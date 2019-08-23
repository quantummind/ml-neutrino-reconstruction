import numpy as np
from mlreco.main_funcs import process_config
from mlreco.iotools.factories import loader_factory
import yaml

cfg = '''
    iotool:
      batch_size: 1
      shuffle: False
      num_workers: 1
      collate_fn: CollateSparse
      sampler:
        name: RandomSequenceSampler
        batch_size: 1
      dataset:
        name: LArCVDataset
        data_dirs:
          - /gpfs/slac/staas/fs1/g/neutrino/kterao/data/dlprod_ppn_v10/combined
        data_key: train_512px
        limit_num_files: 10
        schema:
          dbscan_label:
            - parse_dbscan
            - sparse3d_fivetypes
          group_label: 
            - parse_cluster3d_clean
            - cluster3d_mcst
            - sparse3d_fivetypes
          em_primaries:
            - parse_em_primaries
            - sparse3d_data
            - particle_mcst
    model:
      name: full_edge_model
      modules:
        edge_model:
          name: full_nnconv
          model_cfg:
              leak: 0.1
          balance_classes: True
          loss: 'CE'
          model_path: '/gpfs/slac/staas/fs1/g/neutrino/qatom/gnn-models/early_stop_pi0-20.ckpt'
      network_input:
        - dbscan_label
        - em_primaries
      loss_input:
        - dbscan_label
        - group_label
        - em_primaries
    training:
      seed: 0
      learning_rate: 0.0025
      gpus: '0'
      weight_prefix: '/gpfs/slac/staas/fs1/g/neutrino/qatom/gnn-models/early_stop_pi0-20.ckpt'
      iterations: 1000
      report_step: 1
      checkpoint_step: 100
      log_dir: logs/edge_gnn/edge_node_only
      model_path: ''
      train: True
      debug: False
      minibatch_size: 1
    '''

cfg = yaml.load(cfg,Loader=yaml.Loader)
process_config(cfg)
loader, cfg['data_keys'] = loader_factory(cfg)

import torch
import pickle
from train_voxel_gnn import GraphDataset

from os import listdir
from torch.utils.data import DataLoader
from mlreco.main_funcs import cycle

data_path = '/gpfs/slac/staas/fs1/g/neutrino/qatom/gnn_pi0_reco_nocompton/'
valid_data = GraphDataset(data_path, 1, data_key='00000')
valid_dataset = cycle(valid_data)

from mlreco.trainval import trainval
from mlreco.main_funcs import get_data_minibatched

nf_dim, ef_dim = valid_data.get_feature_dimensions()
cfg['model']['modules']['edge_model']['model_cfg']['n_node_features'] = nf_dim
cfg['model']['modules']['edge_model']['model_cfg']['n_edge_features'] = ef_dim
cfg['model']['network_input'] = ['vertex_features', 'edge_features', 'edge_index', 'batch']
cfg['model']['loss_input'] = ['target', 'edge_index', 'batch']
cfg['data_keys'] = ['edge_index', 'vertex_features', 'edge_features', 'batch', 'target']
Trainer = trainval(cfg)
loaded_iteration = Trainer.initialize()

from mlreco.utils.gnn.features.utils import edge_labels_to_node_labels
from mlreco.utils.metrics import *

valid_dataset = cycle(valid_data)
pred_sbd = []
for t in range(len(valid_data)):
    data_blob = get_data_minibatched(valid_dataset, cfg)
    Trainer._train = False
    res = Trainer.forward(data_blob)
    data_blob = get_data_minibatched(valid_dataset, cfg)
    input_keys = Trainer._input_keys
    loss_keys = Trainer._loss_keys
    target = data_blob['target'][0][0]
    positions = data_blob['vertex_features'][0][0][:, :3].numpy()
    edges = data_blob['edge_index'][0][0].numpy().T
    with torch.set_grad_enabled(False):
        res_combined = {}
        for idx in range(len(data_blob['edge_index'])):
            blob = {}
            for key in data_blob.keys():
                blob[key] = data_blob[key][idx]
            for key in blob:
                blob[key] = [torch.as_tensor(d).cuda() for d in blob[key]]
            data = []
            data.append([blob[key][0] for key in input_keys])
            result = Trainer._net(data)

    #     for key in data_blob:
    #         data_blob[key] = [torch.as_tensor(d) for d in data_blob[key][0]]
    #     data = []
    #     data.append([data_blob[key][0] for key in input_keys])
    #     data = data[0]
    #     print('data', data)
    #     result = Trainer._net(data)[0][0]
            result[0][0][:, 1] += 0.8
            _, pred_inds = torch.max(result[0][0], 1)
            pred_inds = pred_inds.cpu().numpy()
    
    node_labels_truth = edge_labels_to_node_labels(edges, target)
    node_labels_pred = edge_labels_to_node_labels(edges, pred_inds)
    
#     node_labels_truth = merge_labels(positions, node_labels_truth)
    node_labels_pred = merge_labels(positions, node_labels_pred)
    
    pred_sbd.append(SBD(node_labels_pred, node_labels_truth))
#     print(pred_sbd[-1])
    if t % 10 == 0:
        print('prediction SBD', round(np.mean(pred_sbd), 2), '±', round(np.std(pred_sbd)/np.sqrt(len(pred_sbd)), 2), sep='\t')
  


"""
EXAMPLES for plotting:

from mlreco.utils.gnn.features.utils import edge_labels_to_node_labels
# valid_dataset = cycle(valid_data)
data_blob = get_data_minibatched(valid_dataset, cfg)
Trainer._train = False
res = Trainer.forward(data_blob)
data_blob = get_data_minibatched(valid_dataset, cfg)
input_keys = Trainer._input_keys
loss_keys = Trainer._loss_keys
target = data_blob['target'][0][0]
positions = data_blob['vertex_features'][0][0][:, :3].numpy()
edges = data_blob['edge_index'][0][0].numpy().T
with torch.set_grad_enabled(False):
    res_combined = {}
    for idx in range(len(data_blob['edge_index'])):
        blob = {}
        for key in data_blob.keys():
            blob[key] = data_blob[key][idx]
        for key in blob:
            blob[key] = [torch.as_tensor(d).cuda() for d in blob[key]]
        data = []
        data.append([blob[key][0] for key in input_keys])
        result = Trainer._net(data)

#     for key in data_blob:
#         data_blob[key] = [torch.as_tensor(d) for d in data_blob[key][0]]
#     data = []
#     data.append([data_blob[key][0] for key in input_keys])
#     data = data[0]
#     print('data', data)
#     result = Trainer._net(data)[0][0]
        result[0][0][:, 1] += 0.8
        _, pred_inds = torch.max(result[0][0], 1)
        pred_inds = pred_inds.cpu().numpy()
node_labels_truth = edge_labels_to_node_labels(edges, target)

node_labels_pred = edge_labels_to_node_labels(edges, pred_inds)


# plot the truth
coords = positions
x, y, z = coords.T
labels = node_labels_truth
color = np.zeros(len(labels))
i = 0
choices = np.unique(labels)
np.random.shuffle(choices)
for u in choices:
    color[np.where(labels == u)] = i
    i += 1
xyz_range = ((np.min(x)-50, np.max(x)+50), (np.min(y)-50, np.max(y)+50), (np.min(z)-1, np.max(z)+50))
seg_data = trace(x, y, z, color=color, colorscale='Jet', markersize=3, hovertext=color)
plot([seg_data], xyz_range)


# plot the GNN output
coords = positions
x, y, z = coords.T
labels = node_labels_pred
color = np.zeros(len(labels))
i = 0
choices = np.unique(labels)
np.random.shuffle(choices)
for u in choices:
    color[np.where(labels == u)] = i
    i += 1
xyz_range = ((np.min(x)-50, np.max(x)+50), (np.min(y)-50, np.max(y)+50), (np.min(z)-1, np.max(z)+50))
seg_data = trace(x, y, z, color=color, colorscale='Jet', markersize=3, hovertext=color)
plot([seg_data], xyz_range)



# post-process the GNN output to merge labels that overlap significantly
from sklearn.neighbors import NearestNeighbors
import itertools

def merge_labels(positions, labels):
    # merge spectral clusters with directly neighboring voxels
    coords = positions
    labels = labels.copy()
    neigh = NearestNeighbors(n_neighbors=6, radius=1.0)
    neigh.fit(coords)
    in_radius = neigh.radius_neighbors(coords)[1]

    labels_to_merge = []
    candidate_mergers = []
    for point in in_radius:
        sp_labels = np.unique(labels[point])
        if len(sp_labels) > 1:
            merge = sp_labels.tolist()
            candidate_mergers.extend(list(itertools.combinations(merge, 2)))
    candidate_mergers = np.array(candidate_mergers)
    if len(candidate_mergers) > 0:
        pairs, counts = np.unique(candidate_mergers, axis=0, return_counts=True)
        pairs = pairs[np.where(counts > 20)]
        for merge in pairs:
            merge_index = -1
            for i in range(len(labels_to_merge)):
                for m in merge:
                    if m in labels_to_merge[i]:
                        merge_index = i
                        break
                if merge_index > -1:
                    break
            if merge_index == -1:
                labels_to_merge.append(merge)
            else:
                labels_to_merge[merge_index] = list(set().union(labels_to_merge[merge_index], merge))

        for merge in labels_to_merge:
            for i in range(1, len(merge)):
                labels[np.where(labels == merge[i])] = merge[0]
    return labels

coords = positions
x, y, z = coords.T
labels = merge_labels(positions, labels)
color = np.zeros(len(labels))
i = 0
choices = np.unique(labels)
np.random.shuffle(choices)
for u in choices:
    color[np.where(labels == u)] = i
    i += 1
xyz_range = ((np.min(x)-50, np.max(x)+50), (np.min(y)-50, np.max(y)+50), (np.min(z)-1, np.max(z)+50))
seg_data = trace(x, y, z, color=color, colorscale='Jet', markersize=3, hovertext=color)
plot([seg_data], xyz_range)
"""