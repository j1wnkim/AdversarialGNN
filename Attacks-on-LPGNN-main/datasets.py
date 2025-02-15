import os
from functools import partial
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import ToSparseTensor, RandomNodeSplit
from torch_geometric.utils import to_undirected, to_dense_adj, add_self_loops, subgraph
import torch_geometric.utils as torch_geom_utils
from sklearn.metrics.pairwise import cosine_similarity

from transforms import Normalize, FilterTopClass
# from main import AttackMode
from utils import Enum

import heapq
import numpy as np
import random

from attacks import AttackMode, average_feature_difference


    
class SurrogateDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class KarateClub(InMemoryDataset):
    url = 'https://raw.githubusercontent.com/benedekrozemberczki/karateclub/master/dataset/node_level'
    available_datasets = {
        'twitch',
        'facebook',
        'github',
        'deezer',
        'lastfm',
        'wikipedia'
    }

    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name.lower()
        assert self.name in self.available_datasets

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.name, 'raw')

    @property
    def raw_file_names(self):
        return ['edges.csv', 'features.csv', 'target.csv']

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name, 'processed')

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        for part in ['edges', 'features', 'target']:
            download_url(f'{self.url}/{self.name}/{part}.csv', self.raw_dir)

    def process(self):
        target_file = os.path.join(self.raw_dir, self.raw_file_names[2])
        y = pd.read_csv(target_file)['target']
        y = torch.from_numpy(y.to_numpy(dtype=int))
        num_nodes = len(y)

        edge_file = os.path.join(self.raw_dir, self.raw_file_names[0])
        edge_index = pd.read_csv(edge_file)
        edge_index = torch.from_numpy(edge_index.to_numpy()).t().contiguous()
        edge_index = to_undirected(edge_index, num_nodes)  # undirected edges

        feature_file = os.path.join(self.raw_dir, self.raw_file_names[1])
        x = pd.read_csv(feature_file).drop_duplicates()
        x = x.pivot(index='node_id', columns='feature_id', values='value').fillna(0)
        x = x.reindex(range(num_nodes), fill_value=0)
        x = torch.from_numpy(x.to_numpy()).float()

        data = Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return f'KarateClub-{self.name}()'


supported_datasets = {
    'cora': partial(Planetoid, name='cora'),
    'pubmed': partial(Planetoid, name='pubmed'),
    'facebook': partial(KarateClub, name='facebook'),
    'lastfm': partial(KarateClub, name='lastfm', transform=FilterTopClass(10)),
}


def create_surrogate_dataset(data, original_data):
    
    edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.num_nodes)
    
    dense_adj = to_dense_adj(edge_index)[0]
    
    surrogate_features = []
    labels = []
    sims = []
    
    for node_idx in range(data.num_nodes):
        neighbors = dense_adj[node_idx].nonzero(as_tuple=True)[0]
        neighbor_features = data.x[neighbors]
        
#         print(node_idx)
#         print(neighbors)
#         print(neighbor_features.shape)
#         print("------------------------------")
        
        if neighbor_features.shape[0] == 1:
            continue
        
        variance_vector = torch.var(neighbor_features, dim=0)
        
        mean_vector = torch.mean(neighbor_features, dim=0)
        
        cosine_sim = cosine_similarity(
            mean_vector.unsqueeze(0),
            original_data.x[node_idx].unsqueeze(0)
        )
        
        avg_diff = average_feature_difference(mean_vector, original_data.x[node_idx])
        
        sims.append(cosine_sim)
        
#         if avg_diff < 1:
#             print(avg_diff)
#             print( mean_vector.unsqueeze(0))
#             print(original_data.x[node_idx].unsqueeze(0))
        
        label = 1 if cosine_sim > 0 else 0
        
        
        surrogate_features.append(torch.cat((variance_vector, torch.tensor([len(neighbors)], dtype=torch.float32))))
        labels.append(label)
    
    surrogate_features = torch.stack(surrogate_features)
    labels = torch.tensor(labels, dtype=torch.float32)
#     print(f'The mean cosine similarity was {np.mean(sims)}\n')
    
    return SurrogateDataset(surrogate_features, labels)


def create_subdatasets(data, original_data):
    
    train_features = data.x[data.train_mask]
    train_labels = data.y[data.train_mask]
    
    train_edge_index, _ = subgraph(data.train_mask.nonzero(as_tuple=False).view(-1),
                                  data.edge_index, relabel_nodes=True)
    train_data = Data(x=train_features, edge_index=train_edge_index, y=train_labels)
    
    non_train_mask = ~data.train_mask
    non_train_features = data.x[non_train_mask]
    non_train_labels = data.y[non_train_mask]
    
    non_train_edge_index, _ = subgraph(non_train_mask.nonzero(as_tuple=False).view(-1),
                                      data.edge_index, relabel_nodes=True)
    
    non_train_data = Data(x= non_train_features, edge_index=non_train_edge_index, y=non_train_labels)
    
    original_non_train_features = original_data.x[non_train_mask]
    original_non_train_labels = original_data.y[non_train_mask]
    
    original_non_train_edge_index, _ = subgraph(non_train_mask.nonzero(as_tuple=False).view(-1),
                                      original_data.edge_index, relabel_nodes=True)
    
    original_non_train_data = Data(x= original_non_train_features, edge_index=original_non_train_edge_index, y=original_non_train_labels)
    
    surrogate_data = create_surrogate_dataset(non_train_data, original_non_train_data)
    
    return original_non_train_data, surrogate_data

def load_dataset(
        dataset:        dict(help='name of the dataset', option='-d', choices=supported_datasets) = 'cora',
        data_dir:       dict(help='directory to store the dataset') = './datasets',
        data_range:     dict(help='min and max feature value', nargs=2, type=float) = (0, 1),
        val_ratio:      dict(help='fraction of nodes used for validation') = .25,
        test_ratio:     dict(help='fraction of nodes used for test') = .25,
        attack:         dict(help='attack mode or not') = False,
        attack_type:    dict(help='type of attack') = AttackMode.ADDNODES
        ):
    data = supported_datasets[dataset](root=os.path.join(data_dir, dataset))
    
    if not attack:
#         data = AddTrainValTestMask(split='train_rest', num_val=val_ratio, num_test=test_ratio)(data[0])
        data = RandomNodeSplit(split='train_rest', num_val=val_ratio, num_test=test_ratio)(data[0])
        data = ToSparseTensor()(data)
        data.name = dataset
        data.num_classes = int(data.y.max().item()) + 1

        if data_range is not None:
            low, high = data_range
            data = Normalize(low, high)(data)

        return data
    if attack and attack_type == AttackMode.SHADOW: 
        return data, data[0] # just return the data itself.
    
    
    if attack and attack_type == AttackMode.ADDNODES:
            
        degrees = {}

        node_degrees = torch.zeros(data.x.size(0), dtype=torch.int64)
        node_degrees.scatter_add_(0, data.edge_index[0], torch.ones_like(data.edge_index[0]))
        
        
        fraction = 0.1
        total_nodes = data.x.size(0)
        
        m = int(fraction*total_nodes)
        
        _, top_m_indices = torch.topk(-node_degrees, m, sorted=True)

        num_node_features = data.x.size(1)
        if hasattr(data, 'y'):
            num_classes = torch.unique(data.y).numel()
        else:
            num_classes = None  # or set a default if you know the number of classes


        print(f'Total nodes are {total_nodes}\n')


        # Number of new nodes
#         m = 500

        # Generate random features
        new_x = torch.randn(m, num_node_features)

        # Generate random labels (assuming labels are categorical from 0 to num_classes-1)
        if num_classes:
            new_y = torch.randint(0, num_classes, (m,))

        # Top 500 nodes with the highest degree (assuming you have already computed this)
#             top_m_indices = torch.topk(node_degrees, m, largest=True).indices

        # Create new edges for each of the m new nodes to connect to corresponding top m nodes
        source_nodes = torch.arange(data.x.size(0), data.x.size(0) + m)
        target_nodes = top_m_indices

        new_edge_index = torch.stack([source_nodes, target_nodes], dim=0)

        # Update the graph
        data.x = torch.cat([data.x, new_x], dim=0)  # Concatenate features
        if num_classes:
            data.y = torch.cat([data.y, new_y])  # Concatenate labels if they exist
        data.edge_index = torch.cat([data.edge_index, new_edge_index], dim=1)  # Concatenate edge indices
        
        data = RandomNodeSplit(split='train_rest', num_val=val_ratio, num_test=test_ratio)(data[0])
        
        data.name = dataset
        data.num_classes = int(data.y.max().item()) + 1
        print(data.x[0][0:10])

        if data_range is not None:
            low, high = data_range
            data = Normalize(low, high)(data)
        
        data = ToSparseTensor()(data)
        
        return data
        
    
    data = RandomNodeSplit(split='train_rest', num_val=val_ratio, num_test=test_ratio)(data[0])
    surrogate_data = None
    
    if attack and (attack_type == AttackMode.INFERENCE or attack_type == AttackMode.ADDNODES):
#         data, surrogate_data = create_subdatasets(data)
#         print(data)
#         print(surrogate_data)
        pass
    else:    
        data = ToSparseTensor()(data)
        
    data.name = dataset
    data.num_classes = int(data.y.max().item()) + 1
    print(data.x[0][0:10])
        
    if data_range is not None:
        low, high = data_range
        data = Normalize(low, high)(data)
    
    # CODE ADDED BELOW
    if attack:
        
        
        if attack_type == AttackMode.FLIPNODES:
    
            degrees = {}

            edge_index = data.adj_t.coalesce()

            row, col, _ = edge_index.t().coo()

            node_degrees = torch.zeros(edge_index.size(0), dtype=torch.int64)
            node_degrees.scatter_add(0, row, torch.ones_like(row))
            
            fraction = 0.5
            total_nodes = data.x.size(0)
            print(f'Total nodes are {total_nodes}\n')
        
            m = int(fraction*total_nodes)

            _, top_m_indices = torch.topk(node_degrees, m, sorted=True)

            # Assuming 'adj_t' is the sparse tensor after ToSparseTensor
    #         degree_tensor = data.adj_t.sum(dim=0) + data.adj_t.sum(dim=1)  # Efficient degree calculation using sparse summation
    #         node_degrees = degree_tensor.to_dense().numpy()  # Convert to dense for further processing

    #         sorted_indices = node_degrees.argsort(-1)
    #         m = 0.1  # Assuming you want to select the top 20% nodes
    #         num_nodes_to_select = int(data.num_nodes * m)
    #         top_m_indices = sorted_indices[:500]

            mini = np.inf
            maxi = -np.inf
            poss = []

            for label in data.y:
                mini = min(mini, label.item())
                maxi = max(maxi, label.item())
                if label.item() not in poss:
                    poss.append(label.item()) 

            for node_idx in top_m_indices:
                poss = list(range(mini, maxi+1))
                curr_label = data.y[node_idx].item()
                poss.remove(curr_label)
                data.y[node_idx].fill_(random.choice(poss))
    #             data.x[node_idx, :] =  # make your desired changes to the node features here
        
#         elif attack_type == AttackMode.INFERENCE:
            
        
        # -----------------------------------



#         for val in data.edge_index[0]:
#             if val.item() not in degrees.keys():
#                 degrees[val.item()] = 1
#             else:
#                 degrees[val.item()] += 1

#         items = [(edges, index) for index, edges in degrees.items()]

#         top_m_items = heapq.nlargest(500, items)


#         top_m_ids = [item[1] for item in top_m_items]



#         for index in top_m_ids:
#             poss = list(range(mini, maxi+1))
#             curr_label = data.y[index].item()
#             poss.remove(curr_label)
#             data.y[index].fill_(random.choice(poss))

    #         print(length)
    #         print("The tensor is ", data.y[index])

    #     torch.set_printoptions(threshold=10_000)
    #     print(data.x[0])

    # CODE ADDED ABOVE
    
    
#     data = ToSparseTensor()(data)
#     data.name = dataset
#     data.num_classes = int(data.y.max().item()) + 1
    

    
#     node_features = data.x
#     node_labels = data.y
    
#     print("Node features shape:\n\n", node_features.shape, "\n\n")
#     print("Node labels shape:\n\n", node_labels.shape, "\n\n")
    # --------------------------------------------------------------------------
    
#     node_degrees = torch_geom_utils.degree(data.edge_index[0])
    
#     sorted_degrees, sorted_indices = torch.sort(node_degrees, descending=True)
#     top_n_nodes = sorted_indices[:5]

#     # Access features and labels of the top nodes (if applicable)
#     top_n_features = data.x[top_n_nodes]
#     top_n_labels = data.y[top_n_nodes]

#     print(f"Top {5} nodes with highest degrees: {top_n_nodes}")
#     print(f"Their features: {top_n_features}")
#     # Print labels if available
#     if data.has_y:
#         print(f"Their labels: {top_n_labels}")

    return data
