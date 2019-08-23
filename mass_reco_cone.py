import numpy as np
import pandas as pd
import yaml
import math
from scipy.spatial import distance_matrix


URESNET_ENERGY_MODEL_WEIGHTS = '/gpfs/slac/staas/fs1/g/neutrino/qatom/gu-models/early_stop-fivetypes-pi0-augment-undertrained.mdl'
OUTPUT_DIR = '~/src/pi0-final/nn-corrected-cone-2/'

from os import listdir

from mlreco.iotools import factories
from pi0.utils import point_selection, gamma_direction, gamma2_selection, pi0_pi_selection, cone_clusterer, spectral_clusterer
from osf.particle_api import particle_reader
from process_data import process_data




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
      em_primaries:
        - parse_em_primaries
        - sparse3d_data
        - particle_mcst
"""
cfg=yaml.load(cfg,Loader=yaml.Loader)


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
    
    out, groups, _, _ = process_data(input_true, input_reco, segment_label, group_label)
    return out, groups

import torch
import torch.nn as nn
import sparseconvnet as scn

class UResNet(nn.Module):
    def __init__(self, dimension=3, size=1536, nFeatures=16, depth=5, nClasses=1):
        super(UResNet, self).__init__()
        self.dimension = dimension
        self.size = size
        self.nFeatures = nFeatures
        self.depth = depth
        self.nClasses = nClasses
        reps = 2  # Conv block repetition factor
        kernel_size = 2  # Use input_spatial_size method for other values?
        m = nFeatures  # Unet number of features
        nPlanes = [i*m for i in range(1, depth+1)]  # UNet number of features per level
        # nPlanes = [(2**i) * m for i in range(1, num_strides+1)]  # UNet number of features per level
        nInputFeatures = 6
        self.sparseModel = scn.Sequential().add(
           scn.InputLayer(dimension, size, mode=3)).add(
           scn.SubmanifoldConvolution(dimension, nInputFeatures, m, 3, False)).add( # Kernel size 3, no bias
           scn.UNet(dimension, reps, nPlanes, residual_blocks=True, downsample=[kernel_size, 2])).add(  # downsample = [filter size, filter stride]
           scn.BatchNormReLU(m)).add(
           scn.OutputLayer(dimension))
#         self.linear = torch.nn.Linear(m, nClasses)
        self.linear = torch.nn.Sequential(torch.nn.Linear(m, m//2),
                                         torch.nn.ReLU(0.1),
                                         torch.nn.Linear(m//2, nClasses))

    def forward(self, x):
        """
        x is scn coordinate, feature input
        """
        x = self.sparseModel(x)
        x = self.linear(x)
        return x

import torch.optim as optim
import random
import os

model = UResNet()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.cuda.set_device(0)
device = torch.device("cuda:0")
model.to(device)
model.eval()
model.load_state_dict(torch.load(URESNET_ENERGY_MODEL_WEIGHTS), strict=False)

def worker(data, i):
    if len(np.where(data['segment_label'] == 2)[0]) < 1000:
        return None
    
    
    true_energy_fitted_voxels = []
    charge_fitted_voxels = []
    fitted_energy_fitted_voxels = []
    fitted_energy_recorded_voxels = []
    charge_recorded_voxels = []
    true_energy_recorded_voxels = []
    true_energy_all_voxels = []
    
    pi0_cos = [] # cos theta of gamma pair
    gamma_sep = np.array([]) # minimum backward separation of gammas
    chosen_particles = [] # selected particle indices
    
    d, true_shower_hits = process_data(data)
    
    energies = model((torch.Tensor(d[:, :4]).cuda(), torch.Tensor(d[:, 4:-1]).cuda())).detach().cpu().numpy().flatten()
    em_primaries = data['em_primaries']
    input_true = data['input_true']
    group_label = data['group_label']
    
    # assemble group labels to from primaries
    distances = distance_matrix(em_primaries[:, :3], d[:, :3])
    min_indices = np.argmin(distances, axis=1)
    primary_groups = true_shower_hits[min_indices]
    
    
    # determine shower directions
    gamma_dir, gamma_pca_data, gamma_pca_nhits = gamma_direction.do_calculation(d[:, :3], em_primaries, 
                                                                                radius=16.0, eps=7.6, min_samples=5.8)
    if not len(gamma_dir) or not any(gamma_dir[:,4]):
        print('gamma_dir failure')
        return None
    
    # pair up gamma candidates
    selected_showers, sep_matrix = gamma2_selection.do_iterative_selection(gamma_dir, maximum_sep=10.)
    if not len(selected_showers):
        print('gamma2 failure')
        return None
    gamma_sep = sep_matrix[np.triu_indices(len(selected_showers), k=1)]
    
    # calculate pi0 parameters for gamma pairs
    paired_gammas_mask = selected_showers[:,-1] != 0
    gamma_pairs = np.unique(selected_showers[paired_gammas_mask,-1])
    if not len(gamma_pairs) == 1:
        print('pairlen failure')
        return None
    vtx_data = np.empty((len(gamma_pairs),5))
    
    particles = em_primaries[:, -1]
    
    # find shower hits
#     fitted_shower_hits = []
#     fitted_shower_primary_labels = []
    
    
    fitted_shower_hits = cone_clusterer.cluster(d[:, :3], em_primaries[:, :3], params=[50.0, 32.538603965969806, 4.920066409426372, 9.34588243103269], inclusive=True)
    
#     labels, hits = cone_clusterer.cluster(d[:, :3], em_primaries[:, :3], params=[50.0, 32.538603965969806, 4.920066409426372, 9.34588243103269])
#     print('EM primary labels', labels)
#     print('EM primary groups', primary_groups)
#     fitted_shower_hits.append(hits)
#     fitted_shower_primary_labels.append(labels)
#     if not len(fitted_shower_hits[-1]):
#         print('cone failure')
#         return None
    
#     fitted_shower_hits.append(spectral_clusterer.cluster(d[:, :3], em_primaries[:, :3], params=[46.37086851922889, -1.5574991699405842, 0.7537768189993856, 0.9695745937212652]))
#     fitted_shower_primary_labels.append(fitted_shower_hits[-1][min_indices])
#     if not len(fitted_shower_hits[-1]):
#         print('spectral failure')
#         return None
    
    
    for idx,label in enumerate(gamma_pairs):
        gamma_label_mask = selected_showers[:,-1] == label
        gamma_pair = gamma_dir[gamma_label_mask]
        
        gamma0_idx = int(np.argwhere(gamma_label_mask)[0])
        true_gamma0_hits = np.where(true_shower_hits == primary_groups[gamma0_idx])
        gamma1_idx = int(np.argwhere(gamma_label_mask)[1])
        true_gamma1_hits = np.where(true_shower_hits == primary_groups[gamma1_idx])
        print('gamma indices', gamma0_idx, gamma1_idx)

        if len(fitted_shower_hits[gamma0_idx]) == 0 or len(fitted_shower_hits[gamma1_idx]) == 0:
            continue
        
        cos_val = np.dot(gamma_pair[0,-3:], gamma_pair[1,-3:])/np.linalg.norm(gamma_pair[0,-3:])/np.linalg.norm(gamma_pair[1,-3:])
        if cos_val > 1:
            cos_val = 1.0
        pi0_cos += [cos_val]
        
        
        true_energy_fitted_voxels += [[np.sum(d[fitted_shower_hits[gamma0_idx], -1]), np.sum(d[fitted_shower_hits[gamma1_idx], -1])]]
        charge_fitted_voxels += [[np.sum(d[:,4][fitted_shower_hits[gamma0_idx]]), np.sum(d[:,4][fitted_shower_hits[gamma1_idx]])]]
        fitted_energy_fitted_voxels += [[np.sum(energies[fitted_shower_hits[gamma0_idx]]), np.sum(energies[fitted_shower_hits[gamma1_idx]])]]
        
        fitted_energy_recorded_voxels += [[np.sum(energies[true_gamma0_hits]), np.sum(energies[true_gamma1_hits])]]
        charge_recorded_voxels += [[np.sum(d[true_gamma0_hits, 4]), np.sum(d[true_gamma1_hits, 4])]]
        true_energy_recorded_voxels += [[np.sum(d[true_gamma0_hits, -1]), np.sum(d[true_gamma1_hits, -1])]]
        true_energy_all_voxels += [[np.sum(input_true[np.where(group_label[:, -1] == primary_groups[gamma0_idx]), -1]), np.sum(input_true[np.where(group_label[:, -1] == primary_groups[gamma1_idx]), -1])]]
        
        chosen_particles.append([particles[gamma0_idx], particles[gamma1_idx]])
    
    if len(true_energy_fitted_voxels) == 0:
        return None
    
    all_energies = [true_energy_fitted_voxels, charge_fitted_voxels, fitted_energy_fitted_voxels, fitted_energy_recorded_voxels, charge_recorded_voxels, true_energy_recorded_voxels, true_energy_all_voxels]
    return (all_energies, np.array(pi0_cos), np.array(chosen_particles).astype(int), 
            gamma_sep, gamma_dir[paired_gammas_mask,-3:], gamma_pca_data[paired_gammas_mask,-1], 
            gamma_pca_nhits[paired_gammas_mask,-1])


import pi0.utils.cone_clusterer
import pickle

directory = OUTPUT_DIR

files = sorted(listdir(cfg['iotool']['dataset']['data_dirs'][0]))
e1 = []
e2 = []
e3 = []
e4 = []
e5 = []
e6 = []
e7 = []

l3 = np.zeros(0)
l4 = np.zeros(0)
l5 = np.zeros(0)
l6 = np.zeros(0)
l7 = np.zeros(0) # backward gamma separation
l8 = np.empty((0,3)) # fitted gamma directions
l9 = np.zeros(0) # gamma pca values
l10 = np.zeros(0) # gamma nhits in pca
for f in range(len(files)):
    cfg['iotool']['dataset']['data_key'] = files[f]
    loader,data_keys = factories.loader_factory(cfg)
    it = iter(loader)
    preader = particle_reader(cfg['iotool']['dataset']['data_dirs'][0] + '/' + files[f])
    for i in range(len(loader.dataset)):
#         print(i/len(loader.dataset))
        pinfo = preader.get_event(i)
        out = worker(next(it), f)
        if out is not None and len(out[0][0][0]) > 0:
            all_energies, pi0_cos, particles, gamma_sep, gamma_dir, gamma_pca, gamma_pca_nhits = out
            
            energies = pinfo['creation_energy'][particles]
            true_x = pinfo['direction_x'][particles]
            true_y = pinfo['direction_y'][particles]
            true_z = pinfo['direction_z'][particles]
            true_dirs = np.array([true_x.T, true_y.T, true_z.T]).T
            true_cos = np.zeros(len(true_dirs))
            for i in range(len(true_dirs)):
                true_cos[i] = np.dot(true_dirs[i][0], true_dirs[i][1])/np.linalg.norm(true_dirs[i][0])/np.linalg.norm(true_dirs[i][1])
            
            e1.extend(all_energies[0])
            e2.extend(all_energies[1])
            e3.extend(all_energies[2])
            e4.extend(all_energies[3])
            e5.extend(all_energies[4])
            e6.extend(all_energies[5])
            e7.extend(all_energies[6])
            
            l3 = np.concatenate((l3, energies[:, 0]))
            l4 = np.concatenate((l4, energies[:, 1]))
            l5 = np.concatenate((l5, pi0_cos))
            l6 = np.concatenate((l6, true_cos))
            l7 = np.concatenate((l7, gamma_sep))
            l8 = np.concatenate((l8, gamma_dir), axis=0) # fitted gamma directions
            l9 = np.concatenate((l9, gamma_pca)) # gamma pca values
            l10 = np.concatenate((l10, gamma_pca_nhits)) # gamma nhits in pca
            
            np.save(directory + 'pi0_true_energy_fitted_voxels.npy', e1)
            np.save(directory + 'pi0_charge_fitted_voxels.npy', e2)
            np.save(directory + 'pi0_fitted_energy_fitted_voxels.npy', e3)
            np.save(directory + 'pi0_fitted_energy_recorded_voxels.npy', e4)
            np.save(directory + 'pi0_charge_recorded_voxels.npy', e5)
            np.save(directory + 'pi0_true_energy_recorded_voxels.npy', e6)
            np.save(directory + 'pi0_true_energy_all_voxels.npy', e7)
            
            np.save(directory + 'pi0_true_gamma0.npy', l3)
            np.save(directory + 'pi0_true_gamma1.npy', l4)
            np.save(directory + 'pi0_fitted_cos.npy', l5)
            np.save(directory + 'pi0_true_cos.npy', l6)
            np.save(directory + 'pi0_gamma_sep.npy', l7)
            np.save(directory + 'pi0_gamma_dir.npy', l8)
            np.save(directory + 'pi0_gamma_pca.npy', l9)
            np.save(directory + 'pi0_gamma_pca_nhits.npy', l10)