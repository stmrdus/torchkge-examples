import torch
from torch import nn
from torch import cuda
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_

from torchkge.evaluation import LinkPredictionEvaluator
from torchkge.models.interfaces import Model
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


class ConvKB(Model):
    def __init__(self, num_entities, num_relations, emb_dim=100, n_filters=64):
        super(ConvKB, self).__init__(num_entities, num_relations)
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.emb_dim = emb_dim
        self.n_filters = n_filters

        self.ent_embeddings = nn.Embedding(num_entities, self.emb_dim)
        self.rel_embeddings = nn.Embedding(num_relations, self.emb_dim)
        self.convlayer = nn.Sequential(nn.Conv1d(3, n_filters, 1, stride=1), nn.ReLU())
        self.output = nn.Sequential(nn.Linear(emb_dim * n_filters, 2), nn.Softmax(dim=1))

        xavier_uniform_(self.ent_embeddings.weight.data)
        xavier_uniform_(self.rel_embeddings.weight.data)

    def normalize_parameters(self):
        raise NotImplementedError

    def get_embeddings(self):
        self.normalize_parameters()
        return self.ent_embeddings.weight.data, self.rel_embeddings.weight.data

    def lp_prep_cands(self, h_idx, t_idx, r_idx):
        b_size = h_idx.shape[0]

        h_emb = self.ent_embeddings(h_idx)
        t_emb = self.ent_embeddings(t_idx)
        r_emb = self.rel_embeddings(r_idx)

        candidates = self.ent_embeddings.weight.data.view(1, self.num_entities, self.emb_dim)
        candidates = candidates.expand(b_size, self.num_entities, self.emb_dim)

        return h_emb, t_emb, candidates.view(b_size, self.num_entities, 1, self.emb_dim), r_emb

    def lp_scoring_function(self, h, t, r):
        """Link prediction evaluation helper function. See
        torchkge.models.interfaces.Models for more details on the API.

        """
        b_size = h.shape[0]

        if len(h.shape) == 2:
            concat = torch.cat(
                (h.view(b_size, 1, self.emb_dim), r.view(b_size, 1, self.emb_dim)), dim=1)
            concat = concat.view(b_size, 1, 2, self.emb_dim)
            concat = concat.expand(b_size, self.n_ent, 2, self.emb_dim)
            concat = torch.cat((concat, t), dim=2)
            # shape = (b_size, n_ent, 3, emb_dim)
            concat = concat.reshape(-1, 3, self.emb_dim)

        else:
            concat = torch.cat((r.view(b_size, 1, self.emb_dim), t.view(b_size, 1, self.emb_dim)), dim=1)
            concat = concat.view(b_size, 1, 2, self.emb_dim)
            concat = concat.expand(b_size, self.n_ent, 2, self.emb_dim)
            concat = torch.cat((h, concat), dim=2)
            # shape = (b_size, n_entities, 3, emb_dim)
            concat = concat.reshape(-1, 3, self.emb_dim)

        scores = self.output(self.convlayer(
            concat).reshape(concat.shape[0], -1))
        scores = scores.reshape(b_size, -1, 2)
        return scores[:, :, 1]

    def scoring_function(self, h, t, r):
        b_size = h.shape[0]
        h = self.ent_embeddings(h).view(b_size, 1, -1)
        t = self.ent_embeddings(t).view(b_size, 1, -1)
        r = self.rel_embeddings(r).view(b_size, 1, -1)
        concat = torch.cat((h, r, t), dim=1)
        return self.output(self.convlayer(concat).reshape(b_size, -1))

    def forward(self, h, t, nh, nt, r):
        return self.scoring_function(h, t, r), self.scoring_function(nh, nt, r)


class SoftplusLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.Softplus()

    def forward(self, positive_scores, negative_scores):
        return (self.criterion(-positive_scores).mean() + self.criterion(negative_scores).mean()) / 2


# Define model
model = ConvKB(kg_train.n_ent, kg_train.n_rel, emb_dim=100, n_filters=32)

# Define criterion for training model
criterion = SoftplusLoss()


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
