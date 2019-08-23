import numpy as np

OUTPUT_DATASET = '/gpfs/slac/staas/fs1/g/neutrino/qatom/gnn_pi0_reco_nocompton/'
BLUR_DATA = False

from mlreco.main_funcs import process_config
import yaml
from mlreco.iotools.factories import loader_factory
import pickle
from os import listdir
from mlreco.main_funcs import cycle
from mlreco.utils.gnn.features.core import generate_graph, generate_truth
from scipy.spatial.qhull import QhullError

from process_data import process_data

io_cfg = {
    "batch_size": 1,
    "shuffle": False,
    "num_workers": 1,
    "collate_fn": "CollateSparse",
    "dataset": {
        "name": "LArCVDataset",
        "data_dirs": ["/gpfs/slac/staas/fs1/g/neutrino/kvtsang/pions/train"],
        "data_key": "larcv",
#         "data_dirs": ["/gpfs/slac/staas/fs1/g/neutrino/kterao/data/dlprod_ppn_v10/combined"],
#         "data_key": "train_512px",
        "limit_num_files": 999999999,
        "schema": {
            "group_label": ["parse_cluster3d_clean", "cluster3d_mcst", "sparse3d_fivetypes"],
            "segment_label": ["parse_sparse3d_scn", "sparse3d_fivetypes"],
            "em_primaries": ["parse_em_primaries", "sparse3d_data", "particle_mcst"],
            "input_true": ["parse_sparse3d_scn", "sparse3d_data"],
            "input_reco": ["parse_sparse3d_scn", "sparse3d_reco"]
        }
    }
}

cfg = {
    "iotool": io_cfg,
}


start_num = 0
if len(sys.argv) == 2:
    start_num = int(sys.argv[1])
    

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
    
    _, _, found_data, chosen_indices = process_data(input_true, input_reco, segment_label, group_label)
    
    newdata = {}
    newdata['input_true'] = found_data
    newdata['segment_label'] = segment_label[chosen_indices]
    newdata['group_label'] = group_label[chosen_indices]
    newdata['em_primaries'] = data['em_primaries']
    
    return newdata

def worker(d, fname, compton_cut=30):
    print(len(np.where(d['segment_label'] == 2)[0]))
    if len(d['em_primaries']) == 0:
        print('no primaries for', fname)
        return None
    elif len(np.where(d['segment_label'] == 2)[0]) < 1000:
        print('no/too little EM shower for', fname)
        return None
    
    if len(d['em_primaries']) == 0:
        print('no primaries for', fname)
        return None
    elif len(np.where(d['segment_label'] == 2)[0]) == 0:
        print('no EM shower for', fname)
        return None
    
    # if you want to keep compton clusters, set filter_compton=False
    # if you want to add spectral clustering features, add 'spectral' to feature_types
    positions, edges, nf, ef, groups = generate_graph(d, feature_types=['basic', 'cone', 'dbscan'], filter_compton=True)
    _, _, labels = generate_truth(d, positions, groups, edges)
    contents = (edges, nf, ef, labels)
    pickle.dump(contents, open(fname, 'wb+'))
    print('saved', fname)
    return fname
            
files = sorted(listdir(cfg['iotool']['dataset']['data_dirs'][0]))
orig_key = cfg['iotool']['dataset']['data_key']
worker_inputs = []
for f in range(len(files)):
    if orig_key not in files[f]:
        continue
        
    cfg['iotool']['dataset']['data_key'] = files[f]
    loader, _ = loader_factory(cfg)
    dataset = iter(cycle(loader))
    names = []
    while True:
        d = next(dataset, None)
        name = d['index'][0][0]
        if name in names:
            break
        names.append(name)
        if name < start_num:
            continue
        fname = OUTPUT_DATASET + str(f*1000 + name).zfill(10) + '.pkl'
        if BLUR_DATASET:
            d = process(d)
        try:
            worker(d, fname)
        except QhullError:
            print('QhullError for', fname)
        except ValueError:
            print('Only Compton clusters for', fname)