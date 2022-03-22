import torch
from torch import nn
from torch import cuda
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_

from torchkge.evaluation import LinkPredictionEvaluator
from torchkge.models.interfaces import TranslationModel
from torchkge.sampling import BernoulliNegativeSampler
from torchkge.utils import MarginLoss, DataLoader, load_wn18rr

from tqdm.autonotebook import tqdm

# Load dataset
kg_train, kg_val, kg_test = load_wn18rr()

# Define some hyper-parameters for training
emb_dim = 100
lr = 0.0004
n_epochs = 10
b_size = 32768
margin = 0.5

class BaseTransH(TranslationModel):
    def __init__(self, num_entities, num_relations, dim=100):
        super(BaseTransH, self).__init__(num_entities, num_relations, dissimilarity_type='L2')
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.dim = dim

        self.ent_embeddings = nn.Embedding(num_entities, self.dim)
        self.rel_embeddings = nn.Embedding(num_relations, self.dim)
        self.norm_vect = nn.Embedding(num_relations, self.dim)

        xavier_uniform_(self.ent_embeddings.weight.data)
        xavier_uniform_(self.rel_embeddings.weight.data)

        self.normalize_parameters()

        self.evaluated_projections = False
        self.projected_entities = nn.Parameter(torch.empty(size=(num_relations, num_entities, self.dim)), requires_grad=False)

    def normalize_parameters(self):
        self.ent_embeddings.weight.data = F.normalize(self.ent_embeddings.weight.data, p=2, dim=1)
        self.rel_embeddings.weight.data = F.normalize(self.rel_embeddings.weight.data, p=2, dim=1)
        self.norm_vect.weight.data = F.normalize(self.norm_vect.weight.data, p=2, dim=1)

    def get_embeddings(self):
        self.normalize_parameters()
        return self.ent_embeddings.weight.data, self.rel_embeddings.weight.data

    def lp_prep_cands(self, h_idx, t_idx, r_idx):
        if not self.evaluated_projections:
            self.lp_evaluate_projections()

        r = self.rel_embeddings(r_idx)
        proj_h = self.projected_entities[r_idx, h_idx]
        proj_t = self.projected_entities[r_idx, t_idx]
        proj_candidates = self.projected_entities[r_idx]
        return proj_h, proj_t, proj_candidates, r

    def lp_evaluate_projections(self):
        if self.evaluated_projections:
            return
        for i in tqdm(range(self.num_entities), unit='entities', desc='Projecting entities'):
            norm_vect = self.norm_vect.weight.data.view(self.num_relations, self.dim)
            mask = torch.tensor([i], device=norm_vect.device).long()
            if norm_vect.is_cuda:
                cuda.empty_cache()
            ent = self.ent_embeddings(mask)
            norm_components = (ent.view(1, -1) * norm_vect).sum(dim=1)
            self.projected_entities[:, i, :] = (ent.view(1, -1) - norm_components.view(-1, 1) * norm_vect)
            del norm_components
        self.evaluated_projections = True

    def forward(self, h, t, nh, nt, r):
        return self.scoring_function(h, t, r), self.scoring_function(nh, nt, r)

    @staticmethod
    def project(ent, norm_vect):
        return ent - (ent * norm_vect).sum(dim=1).view(-1, 1) * norm_vect

    @staticmethod
    def l2_dissimilarity(a, b):
        assert len(a.shape) == len(b.shape)
        return (a-b).norm(p=2, dim=-1)**2

    @staticmethod
    def l1_dissimilarity(a, b):
        assert len(a.shape) == len(b.shape)
        return (a-b).norm(p=1, dim=-1)

class TransH(BaseTransH):
    def scoring_function(self, h_idx, t_idx, r_idx):
        self.evaluated_projections = False
        h = F.normalize(self.ent_embeddings(h_idx), p=2, dim=1)
        t = F.normalize(self.ent_embeddings(t_idx), p=2, dim=1)
        r = self.rel_embeddings(r_idx)
        norm_vect = F.normalize(self.norm_vect(r_idx), p=2, dim=1)
        scores = -torch.norm(self.project(h, norm_vect=norm_vect) + r - self.project(t, norm_vect=norm_vect), 2, -1)
        return scores


class MarginLoss(nn.Module):
    def __init__(self, margin):
        super().__init__()
        self.margin = margin
        self.loss = nn.MarginRankingLoss(margin=margin, reduction='mean')

    def forward(self, positive_scores, negative_scores):
        return self.loss(positive_scores, negative_scores, target=torch.ones_like(positive_scores))


# Define model
model = TransH(kg_train.n_ent, kg_train.n_rel, dim=64)

# Define criterion for training model
criterion = MarginLoss(margin=0.5)


# Move everything to CUDA if available
if cuda.is_available():
    cuda.empty_cache()
    model.cuda()
    criterion.cuda()

# Define the torch optimizer to be used
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

# Define negative sampler
sampler = BernoulliNegativeSampler(kg_train)

# Define Dataloader
dataloader = DataLoader(kg_train, batch_size=b_size, use_cuda='all')

# Training loop
iterator = tqdm(range(n_epochs), unit='epoch')
for epoch in iterator:
    running_loss = 0.0
    for i, batch in enumerate(dataloader):
        h, t, r = batch[0], batch[1], batch[2]
        n_h, n_t = sampler.corrupt_batch(h, t, r)

        optimizer.zero_grad()

        # forward + backward + optimize
        pos, neg = model(h, t, n_h, n_t, r)
        loss = criterion(pos, neg)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    iterator.set_description('Epoch {} | mean loss: {:.5f}'.format(epoch + 1, running_loss / len(dataloader)))

model.normalize_parameters()


# Define evaluator
evaluator = LinkPredictionEvaluator(model, kg_test)

# Run evaluator
evaluator.evaluate(b_size=128)

# Show results
print("----------------Overall Results----------------")
print('Hit@10: {:.4f}'.format(evaluator.hit_at_k(k=10)[0]))
print('Hit@3: {:.4f}'.format(evaluator.hit_at_k(k=3)[0]))
print('Hit@1: {:.4f}'.format(evaluator.hit_at_k(k=1)[0]))
print('Mean Rank: {:.4f}'.format(evaluator.mean_rank()[0]))
print('Mean Reciprocal Rank : {:.4f}'.format(evaluator.mrr()[0]))