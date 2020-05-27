import pandas as pd
import numpy as np
import torch
from torchtext.data import Field
from torchtext.data import TabularDataset
from torchtext.data import Iterator, BucketIterator
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import tqdm



tokenize = lambda x: x.split()
TEXT = Field(sequential=True, tokenize=tokenize, lower=True)
LABEL = Field(sequential=False, use_vocab=False)

tv_datafields = [("id", None), # we won't be needing the id, so we pass in None as the field
                 ("comment_text", TEXT), ("toxic", LABEL),
                 ("severe_toxic", LABEL), ("threat", LABEL),
                 ("obscene", LABEL), ("insult", LABEL),
                 ("identity_hate", LABEL)]

trn, vld = TabularDataset.splits(
        path="data", # the root directory where the data lies
        train='train.csv', validation="valid.csv",
        format='csv',
        skip_header=True, # if your csv header has a header, make sure to pass this to ensure it doesn't get proceesed as data!
        fields=tv_datafields)
# tst_datafields = [("id", None), # we won't be needing the id, so we pass in None as the field
#                  ("comment_text", TEXT)
# ]

# tst = TabularDataset(
#         path="data/test.csv", # the file path
#         format='csv',
#         skip_header=True, # if your csv header has a header, make sure to pass this to ensure it doesn't get proceesed as data!
#         fields=tst_datafields)
TEXT.build_vocab(trn)

train_iter, val_iter = BucketIterator.splits(
        (trn, vld), # we pass in the datasets we want the iterator to draw data from
        batch_sizes=(64, 64),
        device=-1, # if you want to use the GPU, specify the GPU number here
        sort_key=lambda x: len(x.comment_text), # the BucketIterator needs to be told what function it should use to group the data.
        sort_within_batch=False,
        repeat=False # we pass repeat=False because we want to wrap this Iterator layer.
)
batch = next(train_iter.__iter__()); batch
# test_iter = Iterator(tst, batch_size=64, device=-1, sort=False, sort_within_batch=False, repeat=False)

class BatchWrapper:
    def __init__(self, dl, x_var, y_vars):
        self.dl, self.x_var, self.y_vars = dl, x_var, y_vars # we pass in the list of attributes for x and y
    
    def __iter__(self):
        for batch in self.dl:
            x = getattr(batch, self.x_var) # we assume only one input in this wrapper
            
            if self.y_vars is not None: # we will concatenate y into a single tensor
                y = torch.cat([getattr(batch, feat).unsqueeze(1) for feat in self.y_vars], dim=1).float()
            else:
                y = torch.zeros((1))
            
            print("\nTHIS IS X: ", x.size())
            print("THIS IS Y: ", y.size())

            yield (x, y)
    
    def __len__(self):
        return len(self.dl)

train_dl = BatchWrapper(train_iter, "comment_text", ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"])
valid_dl = BatchWrapper(val_iter, "comment_text", ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"])
# test_dl = BatchWrapper(test_iter, "comment_text", None)


class SimpleBiLSTMBaseline(nn.Module):
    def __init__(self, hidden_dim, emb_dim=300,
                 spatial_dropout=0.05, recurrent_dropout=0.1, num_linear=1):
        super().__init__() # don't forget to call this!
        self.embedding = nn.Embedding(len(TEXT.vocab), emb_dim)
        self.encoder = nn.LSTM(emb_dim, hidden_dim, num_layers=1, dropout=recurrent_dropout)
        self.linear_layers = []
        for _ in range(num_linear - 1):
            self.linear_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.linear_layers = nn.ModuleList(self.linear_layers)
      #   print('LAYER: ', self.linear_layers.size)
        self.predictor = nn.Linear(hidden_dim, 6)
        print("HIDDEN DIM: ", hidden_dim)
    
    def forward(self, seq):
        hdn, _ = self.encoder(self.embedding(seq))
        feature = hdn[-1, :, :]
        print("FEATURE: ", feature.size())
        for layer in self.linear_layers:
            
            feature = layer(feature)
            
        preds = self.predictor(feature)
        print("THIS IS PREDS #1: ", preds.size())
        return preds

em_sz = 100
nh = 500
nl = 3
model = SimpleBiLSTMBaseline(nh, emb_dim=em_sz); model

opt = optim.Adam(model.parameters(), lr=1e-2)
loss_func = nn.BCEWithLogitsLoss()

epochs = 2

for epoch in range(1, epochs + 1):
    running_loss = 0.0
    running_corrects = 0
    model.train() # turn on training mode
    for x, y in tqdm.tqdm(train_dl): # thanks to our wrapper, we can intuitively iterate over our data!
        opt.zero_grad()

        preds = model(x)
        print("THIS IS PREDS: ", preds.size())
        print("THIS IS Y: ", y.size())
        exit()
        loss = loss_func(preds, y)
        loss.backward()
        opt.step()
        
        running_loss += loss.data * x.size(0)
        
    epoch_loss = running_loss / len(trn)
    
    # calculate the validation loss for this epoch
    val_loss = 0.0
    model.eval() # turn on evaluation mode
    for x, y in valid_dl:
        preds = model(x)
        loss = loss_func(preds, y)
        val_loss += loss.data * x.size(0)

    val_loss /= len(vld)
    print('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}'.format(epoch, epoch_loss, val_loss))





