import torch
from torch import nn
import random
import numpy as np
from src.models.modules import *
from torch_geometric.nn import GATConv


class GTMKT(nn.Module):
    def __init__(self, embedding_dim, num_clases, convs=True):
        super().__init__()
        self.target_cluster_size = 32
        self.embedding_dim = embedding_dim
        self.embeddings_layer = nn.LazyLinear(embedding_dim)
        self.transformer = Transformer(depth=8, heads=8, dim=embedding_dim)
        self.class_token = nn.Embedding(1, embedding_dim)
        self.input_layer = nn.LazyLinear(embedding_dim)
        self.conv1 = GATConv(embedding_dim, embedding_dim)
        self.conv2 = GATConv(embedding_dim, embedding_dim)
        self.node_embedding_layer = nn.Embedding(1, embedding_dim)
        self.to_class = nn.LazyLinear(num_clases)
        self.convs = convs

    # This is the neighbour splitting function
    def generate_clusters(self, num_nodes, adj):
        clusters = np.full(num_nodes, 0)
        unvisited_nodes = set({i for i in range(num_nodes)})
        cluster_id = 0
        while len(unvisited_nodes) != 0:
            neighbs = set({})
            cluster = [random.choice(list(unvisited_nodes))]
            unvisited_nodes.remove(cluster[-1])

            neighbs.update(adj[cluster[-1]])
            neighbs = neighbs.intersection(unvisited_nodes)
            clusters[cluster[-1]] = cluster_id
            cluster_is_full = False
            while not cluster_is_full:
                if len(neighbs) == 0:  # We have no neighbors so we add some random unvisited node
                    if len(unvisited_nodes) == 0:
                        cluster_is_full = True
                        break
                    cluster.append(random.sample(unvisited_nodes, 1)[0])
                    unvisited_nodes.remove(cluster[-1])
                    neighbs.update(adj[cluster[-1]])
                    neighbs = neighbs.intersection(unvisited_nodes)
                    clusters[cluster[-1]] = cluster_id
                    if len(cluster) == self.target_cluster_size:
                        cluster_is_full = True
                        break
                while len(neighbs) > 0:  # We do have neighbors so we add one randomly
                    cluster.append(random.choice(list(neighbs)))

                    neighbs.remove(cluster[-1])
                    unvisited_nodes.remove(cluster[-1])
                    neighbs.update(adj[cluster[-1]])
                    neighbs = neighbs.intersection(unvisited_nodes)
                    clusters[cluster[-1]] = cluster_id
                    if len(cluster) == self.target_cluster_size:
                        cluster_is_full = True
                        break
            cluster_id += 1
        return clusters, cluster_id

    def forward(self, g, adj):
        if g.x == None:  # In case we don't have any Node features we learn an embedding
            x = self.node_embedding_layer(torch.tensor([0]))[0].repeat(g.num_nodes, 1)
        else:
            x = g.x.float()
        clusters, num_clusters = self.generate_clusters(g.num_nodes, adj)
        x = self.input_layer(x)
        if self.convs:
            x = self.conv1(x, g.edge_index)
            x = self.conv2(x, g.edge_index)
        cluster_embeddings = []
        for i in range(num_clusters - 1):
            cluster_elements = np.nonzero(clusters == i)[0]
            cluster_embeddings.append(self.embeddings_layer(x[cluster_elements].view(-1)))

        cluster_elements = np.nonzero(clusters == (num_clusters - 1))[0]
        cluster_embeddings.append(self.embeddings_layer(torch.cat((x[cluster_elements].view(-1),
                                                                   torch.zeros(
                                                                       self.embedding_dim * self.target_cluster_size -
                                                                       x[cluster_elements].view(-1).shape[0])))))

        cluster_embeddings = [self.class_token(torch.zeros(1).type(torch.LongTensor)).view(-1)] + cluster_embeddings
        cluster_embeddings = torch.stack(cluster_embeddings)
        # print("cluster_embeddings:", cluster_embeddings)
        result = self.transformer(
            cluster_embeddings.view((1, cluster_embeddings.shape[0], cluster_embeddings.shape[1])))
        return self.to_class(result[0][0].view(1, -1))
