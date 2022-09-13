"""
ACMS 80770-03: Deep Learning with Graphs
Instructor: Navid Shervani-Tabar
Fall 2022
University of Notre Dame

Homework 1: Programming assignment
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from networkx.algorithms import bipartite
from networkx.generators.random_graphs import erdos_renyi_graph
import copy


# -- Initialize graphs
seed = 30
G = nx.florentine_families_graph()
nodes = G.nodes()

layout = nx.spring_layout(G, seed=seed)

# -- keep a copy of edges in the graph
old_edges = copy.deepcopy(G.edges())

# -- compute jaccard's similarity
def jaccard(u,v,G):
  """
    Compute the Jaccard coefficient of the node pair (n.v) in the graph G.
  """
  union_size = len(set(G[u])|set(G[v]))
  if union_size==0:
    return 0
  inter_size = len(set(G[u])& set(G[v]))
  return inter_size/union_size
n = len(G.nodes())
#jaccard matrix S
S = np.zeros((n,n))
new_edges, metrics = [], []
for i,u in enumerate(G.nodes()):
  for j,v in enumerate(G.nodes()):
    S[i,j] = jaccard(u,v,G)
    if u == 'Ginori':
      new_edges.append((u,v))
      metrics.append(S[i,j])
print('The Jaccard similarity matrix S is')
print(S)

# -- plot Florentine Families graph
nx.draw_networkx_nodes(G, nodelist=nodes, label=nodes, pos=layout, node_size=600)
nx.draw_networkx_edges(G, edgelist=old_edges, pos=layout, edge_color='gray', width=4)

# -- plot edges representing similarity
"""
    This example is randomly plotting similarities between 8 pairs of nodes in the graph. 
    Identify the ”Ginori”
"""
ne = nx.draw_networkx_edges(G, edgelist=new_edges, pos=layout, edge_color=np.asarray(metrics), width=4, alpha=0.7)
plt.colorbar(ne)
plt.axis('off')
plt.show()

