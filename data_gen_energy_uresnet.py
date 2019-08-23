import numpy as np
import yaml

OUTPUT_DATASET = '/gpfs/slac/staas/fs1/g/neutrino/qatom/gu_pi0/'


from os import listdir
from process_data import process_data
from mlreco.iotools import factories

# configure data parsers
cfg = """
iotool:
  batch_size: 1
  shuffle: False
  num_workers: 1
  collate_fn: CollateSparse
  dataset:
    name: LArCVDataset
    data_dirs:
      - /gpfs/slac/staas/fs1/g/neutrino/kvtsang/pions/train
    data_key: larcv
    limit_num_files: 1
    schema:
      input_true:
        - parse_sparse3d_scn
        - sparse3d_data
      input_reco:
        - parse_sparse3d_scn
        - sparse3d_reco
      segment_label:
        - parse_sparse3d_scn
        - sparse3d_fivetypes
      group_label:
        - parse_cluster3d_clean
        - cluster3d_mcst
        - sparse3d_fivetypes
"""
cfg=yaml.load(cfg,Loader=yaml.Loader)


from scipy.spatial import distance_matrix

def process(data):
    """
    data: input_true, input_reco, ghost_label, group_label
    returns: input, output
        input: intersection between reco and true, labeled with reco charge depositions
        output: intersection between reco and true, labeled with adjusted energy depositions
    """
    input_true = data['input_true']
    input_reco = data['input_reco']
    segment_label = data['segment_label']
    group_label = data['group_label']
    
    out, _, _, _ = process_data(input_true, input_reco, segment_label, group_label)
    return out


import pickle

start_num = 0

files = sorted(listdir(cfg['iotool']['dataset']['data_dirs'][0]))
orig_key = cfg['iotool']['dataset']['data_key']
worker_inputs = []
for f in range(len(files)):
    if orig_key not in files[f]:
        continue
#     if int(files[f][-10:-5]) < start_num:
#         continue
        
    cfg['iotool']['dataset']['data_key'] = files[f]
    loader, _ = factories.loader_factory(cfg)
    dataset = iter(loader)
    names = []
    index = 0
    num = 0
    while True:
        d = next(dataset, None)
        if num < start_num:
            continue
        num += 1
        if d is None:
            break
            
        n = d['index'][0][0]
        print(n)
        if n in names:
            break
        names.append(n)
        out = process(d)
        if out is not None:
            fname = OUTPUT_DATASET + files[f].split('.')[0] + '_' + str(index).zfill(6) + '.npy'
            np.save(fname, out)
        index += 1