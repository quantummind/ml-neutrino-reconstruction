import numpy as np

data_path = '/gpfs/slac/staas/fs1/g/neutrino/qatom/gnn_pi0_reco_nocompton/'

from mlreco.main_funcs import process_config
from mlreco.iotools.factories import loader_factory
import yaml

import torch
import pickle
from torch.utils.data import Dataset

class GraphDataset(Dataset):
    def __init__(self, data_path, batch_size, data_key='', anti_data_key=None):
        self.data_path = data_path
        files = sorted(listdir(data_path))
        input_files = []
        for f in files:
            if (data_key in f) and (anti_data_key is None or not (anti_data_key in f)):
                input_files.append(f)
        self.input_files = input_files
        self.batch_size = batch_size
    
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.input_files)
    
    def get_feature_dimensions(self):
        d = self.__getitem__(0)
        nf = d['vertex_features']
        ef = d['edge_features']
        return len(nf[0]), len(ef[0])
    
    def __getitem__(self, index):
        'Generates one sample of data'
        all_edges = []
        all_nf = []
        all_ef = []
        all_labels = []
        all_batch = []
        for i in range(self.batch_size):
            if self.batch_size*index+i >= len(self.input_files):
                break
            # Load data and get label
            edges, nf, ef, labels = pickle.load(open(self.data_path + self.input_files[index+i], 'rb'))
            batch = np.zeros(len(nf)) + i
            edges = torch.tensor(edges.T).long()
            nf = torch.tensor(np.array(nf)).float()
            ef = torch.tensor(np.array(ef)).float()
            labels = torch.tensor(labels).long()
#             labels_2d = np.zeros((len(labels), 2))
#             labels_2d[(np.arange(len(labels)), 1 - labels.astype(int))] = 1
#             labels = torch.tensor(labels_2d).long()
            
            all_edges.append(edges)
            all_nf.append(nf)
            all_ef.append(ef)
            all_labels.append(labels)
            all_batch.append(batch)
        if len(all_edges) > 0:
            all_edges = torch.cat(all_edges, 1)
            all_nf = torch.cat(all_nf, 0)
            all_ef = torch.cat(all_ef, 0)
            all_labels = np.concatenate(all_labels, 0)
            all_batch = np.concatenate(all_batch, 0)
        else:
            raise IndexError
        return {'edge_index':all_edges, 'vertex_features':all_nf, 'edge_features':all_ef, 'batch':all_batch, 'target':all_labels}
    
from os import listdir
from torch.utils.data import DataLoader
from mlreco.main_funcs import cycle

if __name__ == '__main__':
    cfg = '''
    iotool:
      batch_size: 2
      shuffle: False
      num_workers: 2
      collate_fn: CollateSparse
      sampler:
        name: RandomSequenceSampler
        batch_size: 2
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
          model_path: ''
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
      weight_prefix: /gpfs/slac/staas/fs1/g/neutrino/qatom/gnn-models/early_stop_pi0
      iterations: 1000
      report_step: 1
      checkpoint_step: 100
      log_dir: logs/edge_gnn/edge_node_only
      model_path: ''
      train: True
      debug: False
      minibatch_size: 2
    '''
    cfg = yaml.load(cfg,Loader=yaml.Loader)
    print(cfg)
    process_config(cfg)
    loader, cfg['data_keys'] = loader_factory(cfg)

    batch_size = cfg['iotool']['batch_size']
    # cfg['training']['minibatch_size'] = batch_size // len(cfg['training']['gpus'])
    # cfg['iotool']['batch_size'] = batch_size
    train_data = GraphDataset(data_path, batch_size, anti_data_key='00000')
    valid_data = GraphDataset(data_path, 1, data_key='00000')
    train_dataset = cycle(train_data)
    valid_dataset = cycle(valid_data)

    nf_dim, ef_dim = train_data.get_feature_dimensions()
    cfg['model']['modules']['edge_model']['model_cfg']['n_node_features'] = nf_dim
    cfg['model']['modules']['edge_model']['model_cfg']['n_edge_features'] = ef_dim

    from mlreco.trainval import trainval
    from mlreco.main_funcs import get_data_minibatched, cycle, make_directories

    cfg['model']['network_input'] = ['vertex_features', 'edge_features', 'edge_index', 'batch']
    cfg['model']['loss_input'] = ['target', 'edge_index', 'batch']
    cfg['data_keys'] = ['edge_index', 'vertex_features', 'edge_features', 'batch', 'target']
    Trainer = trainval(cfg)
    loaded_iteration = Trainer.initialize()
    # make_directories(cfg, loaded_iteration)

    from mlreco.utils.utils import CSVData

    n_epochs = 100

    valid_losses = []
#     log_file = CSVData('log_trash/train_validate_log_0.csv')
    for f in range(n_epochs):
        train_accuracy = 0
        train_sbd = 0
        train_loss = 0
        train_len = len(train_data) // cfg['iotool']['batch_size']
        for i in range(train_len):
    #         cfg['training']['minibatch_size'] = batch_size / len(cfg['training']['gpus'])
    #         cfg['iotool']['batch_size'] = batch_size
            data_blob = get_data_minibatched(train_dataset, cfg)
            Trainer._train = True
            res = Trainer.train_step(data_blob)
            train_accuracy += res['accuracy']
            train_sbd += res['sbd']
            train_loss += res['loss_seg']
            if i % 5 == 0:
                print(i/train_len, 'sbd', train_sbd/(i+1), 'acc', train_accuracy/(i+1), ', loss', train_loss/(i+1))
        train_sbd /= train_len
        train_accuracy /= train_len
        train_loss /= train_len
        print('train_sbd', train_sbd)
        print('train_accuracy', train_accuracy)
        print('train_loss', train_loss)
        Trainer.save_state(-1*f)
        valid_accuracy = 0
        valid_sbd = 0
        valid_loss = 0
        valid_len = len(valid_data) // cfg['iotool']['batch_size']
        for i in range(valid_len):
    #         cfg['training']['minibatch_size'] = 1
    #         cfg['iotool']['batch_size'] = 1
            data_blob = get_data_minibatched(valid_dataset, cfg)
            Trainer._train = False
            res = Trainer.forward(data_blob)
            valid_accuracy += res['accuracy']
            valid_sbd += res['sbd']
            valid_loss += res['loss_seg']
        valid_sbd /= len(valid_data)
        valid_accuracy /= len(valid_data)
        valid_loss /= len(valid_data)
        print('valid_sbd', valid_sbd)
        print('valid_accuracy', valid_accuracy)
        print('valid_loss', valid_loss)
        valid_losses.append(valid_loss)

        if valid_losses[-1] == min(valid_losses):
            Trainer.save_state(f)

#         log_file.record(['train_sbd', 'train_accuracy', 'train_loss'], [train_sbd, train_accuracy, train_loss])
#         log_file.record(['valid_sbd', 'valid_accuracy', 'valid_loss'], [valid_sbd, valid_accuracy, valid_loss])
#         log_file.write()
#         log_file.flush()
#     log_file.close()