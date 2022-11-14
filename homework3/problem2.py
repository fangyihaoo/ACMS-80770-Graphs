"""
ACMS 80770-03: Deep Learning with Graphs
Instructor: Navid Shervani-Tabar
Fall 2022
University of Notre Dame
Homework 2: Programming assignment
Problem 2
"""
import torch

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from torch import nn
from torch.autograd.functional import jacobian


torch.manual_seed(2022)


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
        # A: n*n, H: n*n, initilized as one hot vector
        # H^{k} = D^{-1/2}*(A+I)*D^{-1/2}*H*W
        # return n*
        n = A.shape[-1]
        D_negative_sqrt = torch.diag(1/(A+torch.eye(n)).sum(dim=0).sqrt())
        H = D_negative_sqrt@(A+torch.eye(n))@D_negative_sqrt@H
        H = self.lin(H)
        return H



class MyModel1(nn.Module):
    """
        model1
        200-100
    """
    def __init__(self, A):
        super(MyModel1, self).__init__()
        # -- initialize layers
        torch.manual_seed(2022)
        self.A = A
        self.conv = GCN(200,100)

    def forward(self, h0):
        x = self.conv(self.A,h0)
        x = x.relu()

        return x


class MyModel2(nn.Module):
    """
        model2
        200-100-50-20
    """
    def __init__(self, A):
        super(MyModel2, self).__init__()
        # -- initialize layers
        torch.manual_seed(2022)
        self.A = A
        self.conv1 = GCN(200,100)
        self.conv2 = GCN(100,50)
        self.conv3 = GCN(50,20)

    def forward(self, h0):
        x = self.conv1(self.A,h0)
        x = x.relu()
        x = self.conv2(self.A,x)
        x = x.relu()
        x = self.conv3(self.A,x)
        x = x.relu()

        return x

class MyModel3(nn.Module):
    """
        model2
        200-100-50-20-20-20
    """
    def __init__(self, A):
        super(MyModel3, self).__init__()
        # -- initialize layers
        torch.manual_seed(2022)
        self.A = A
        self.conv1 = GCN(200,100)
        self.conv2 = GCN(100,50)
        self.conv3 = GCN(50,20)
        self.conv4 = GCN(20,20)
        self.conv5 = GCN(20,20)

    def forward(self, h0):
        x = self.conv1(self.A,h0)
        x = x.relu()
        x = self.conv2(self.A,x)
        x = x.relu()
        x = self.conv3(self.A,x)
        x = x.relu()
        x = self.conv4(self.A,x)
        x = x.relu()
        x = self.conv5(self.A,x)
        x = x.relu()

        return x

"""
    Effective range
"""
# -- Initialize graph
seed = 32
n_V = 200   # total number of nodes
i = 17      # node ID
#i = 27
k = 0       # k-hop
G = nx.barabasi_albert_graph(n_V, 2, seed=seed)

for k in [2,4,6]:
# -- plot graph
  plt.figure()
  plt.title(f'Node V{i}, K={k}')
  layout = nx.spring_layout(G, seed=seed, iterations=400)
  nx.draw(G, pos=layout, edge_color='gray', width=2, with_labels=False, node_size=100)

# -- plot neighborhood
  nodes = nx.single_source_shortest_path_length(G, i, cutoff=k)
  im2 = nx.draw_networkx_nodes(G, nodelist=nodes, label=nodes, pos=layout, node_color='red', node_size=100)

# -- visualize
  plt.colorbar(im2)
  plt.show()
  plt.close()


"""
    Influence score
"""
# -- Initialize the model and node feature vectors
A = nx.adjacency_matrix(G)
A = torch.from_numpy(A.todense()).float()

H = torch.eye(n_V)
k = -1
for model in [MyModel1(A),MyModel2(A),MyModel3(A)]:
  output = model(H)
  Jacob = jacobian(model,H)
#-- Influence sore
  inf_score = []
  for j in range(200):
    J = Jacob[i,:,j:] #100*200*200*200
    inf_score.append(J.sum())
  # -- plot influence scores
  plt.figure()
  k +=2
  plt.title(f'Node V{i}, {k} layers')
  layout = nx.spring_layout(G, seed=seed, iterations=400)
  nx.draw(G, pos=layout, edge_color='gray', width=2, with_labels=False, node_size=100)
  nodes = list(range(200))
  im2 = nx.draw_networkx_nodes(G, nodelist=nodes, label=nodes, pos=layout, node_color=inf_score, node_size=100)
  plt.colorbar(im2)
  plt.show()
  plt.close()
