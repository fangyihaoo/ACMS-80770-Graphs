from pandas.core.groupby.groupby import OutputFrameOrSeries
"""
ACMS 80770-03: Deep Learning with Graphs
Instructor: Navid Shervani-Tabar
Fall 2022
University of Notre Dame
Homework 2: Programming assignment
Problem 1
"""
import torch
from torch import nn
from torch import optim
import warnings
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

warnings.simplefilter(action='ignore', category=UserWarning)
from chainer_chemistry import datasets
from chainer_chemistry.dataset.preprocessors.ggnn_preprocessor import GGNNPreprocessor


"""
    load data
"""
dataset, dataset_smiles = datasets.get_qm9(GGNNPreprocessor(kekulize=True), return_smiles=True,
                                           target_index=np.random.choice(range(133000), 6000, False))

V = 9
atom_types = [6, 8, 7, 9, 1]

def adj(x):
    x = x[1]
    adjacency = np.zeros((V, V)).astype(float)
    adjacency[:len(x[0]), :len(x[0])] = x[0] + 2 * x[1] + 3 * x[2]
    return torch.tensor(adjacency)


def sig(x):
    x = x[0]
    atoms = np.ones((V)).astype(float)
    atoms[:len(x)] = x
    out = np.array([int(atom == atom_type) for atom_type in atom_types for atom in atoms]).astype(float)
    return torch.tensor(out).reshape(5, len(atoms)).T


def target(x):
    x = x[2]
    return torch.tensor(x)


adjs = torch.stack(list(map(adj, dataset))).float()
sigs = torch.stack(list(map(sig, dataset))).float()
prop = torch.stack(list(map(target, dataset)))[:, 5].float()


class GCN:
    """
        Graph convolutional layer
    """
    def __init__(self, in_features, out_features):
        # -- initialize weight
        self.lin = nn.Linear(in_features,out_features)

        # -- non-linearity, here we add relu in model class


    def __call__(self, A, H):
        # -- GCN propagation rule
        # A: batch_size*n*n, H:batch_size*n*5
        # H^{k} = D^{-1/2}*(A+I)*D^{-1/2}*H*W
        # return batch_size*n*3
        n = A.shape[-1]
        D_negative_sqrt = torch.diag_embed(1/(A+torch.eye(n)).sum(dim=2).sqrt())
        H = D_negative_sqrt@(A+torch.eye(n))@D_negative_sqrt@H
        H = self.lin(H)
        return H
        


class GraphPooling:
    """
        Graph pooling layer
    """
    def __init__(self):
        pass

    def __call__(self, H):
        # -- multi-set pooling operator
        # H:batch_size*n*3
        # sum along the sencond axis, return batch_size*3 
        return H.sum(dim=1)
        


class MyModel(nn.Module):
    """
        Regression  model
    """
    def __init__(self):
        super(MyModel, self).__init__()
        # -- initialize layers
        torch.manual_seed(2022)
        self.conv = GCN(5,3)
        self.pool = GraphPooling()
        self.lin = nn.Linear(3,1)

    def forward(self, A, h0):
        # output: batch_size*1
        x = self.conv(A,h0)
        x = x.relu()
        x = self.pool(x)
        x = self.lin(x)

        return x


"""
    Train
"""
# -- Initialize the model, loss function, and the optimizer
model = MyModel()
MyLoss = nn.MSELoss()
MyOptimizer = optim.SGD(model.parameters(),lr=0.001)
loss_epoch = []
batch_size = 100
# -- update parameters
for epoch in range(200):
    cum_loss = 0
    # 50 iterations needed for 5000 samples
    for i in range(50):

        # -- predict
        MyOptimizer.zero_grad()
        pred = model(adjs[i*batch_size:(i+1)*batch_size], sigs[i*batch_size:(i+1)*batch_size])
        labels = prop[i*batch_size:(i+1)*batch_size]

        # -- loss
        loss = MyLoss(pred,labels)
        
        # -- optimize
        loss.backward()
        MyOptimizer.step()
        
        cum_loss += loss.item()
    
    loss_epoch.append(cum_loss/50)
    #print(f'Loss: {cum_loss/10}')


# -- plot loss
X = np.arange(1,201)
fig,ax = plt.subplots(figsize=(10,5))
ax.plot(X, loss_epoch,'r--', lw=2)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.set_xlabel('epoch', fontsize=15)
ax.set_ylabel('MSE Loss', fontsize=15)

# evaluate
pred = model(adjs[5000:], sigs[5000:])
labels = prop[5000:]
pred = pred.detach().numpy()
labels = labels.detach().numpy()
fig,ax = plt.subplots(figsize=(8,8))
ax.scatter(labels, pred,alpha=1.0)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.set_xlabel('Target', fontsize=15)
ax.set_ylabel('Predition', fontsize=15)
ax.set_ylim([-0.40,-0.15]);
ax.set_xlim([-0.40,-0.15]);
