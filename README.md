# Machine Learning Methods for Event Reconstruction with Liquid Argon Time Projection Chamber Data

This repository, in combination with my forks of `lartpc_mlreco3d` and `pi0_reco`, are the result of my Summer Undergraduate Research Fellowship at Stanford SLAC (2019). The work covers two main topics of event reconstruction in liquid argon time projection chamber (LArTPC) data: shower reconstruction (using cone clustering, density-based clustering, spectral methods, and graph neural networks) and energy reconstruction (using semantic segmentation with sparse convolutional neural networks). The two topics are tied together in reconstruction of the neutral pion mass from Monte Carlo simulated data.

## Requirements

My fork of `lartpc_mlreco3d` (develop branch), my fork of `pi0_reco`, and the `dlp_opendata_api`.


## U-ResNet energy reconstruction

U-ResNet regression on charge deposition + segmentation labels to predict energy deposition.

Create data to a temporary directory `OUTPUT_DATASET` with `data_gen_energy_uresnet.py`. Train the U-ResNet with `train_energy_uresnet.py` after setting the model weight output `MODEL_OUTPUT`, the input data directory `data_path`, and the maximum side length of the dataset `DATA_SIZE`. The model is run during pi0 mass reconstruction, but a more detailed analysis of running the model can be found outside this repo in the notebook `/gpfs/slac/staas/fs1/g/neutrino/qatom/run uresnet-charge-to-energy-fivetypes-pi0.ipynb`.


## Graph neural network voxel-by-voxel shower clustering

Message-passing graph neural network for clustering of showers by edge classification.

Generate graphs from LArTPC events with `data_gen_voxel_gnn.py` after setting `OUTPUT_DATASET`. Train the GNN with `train_voxel_gnn.py` after setting the input data directory `data_path` and the model weight output in the `cfg` string. Run the GNN and get an estimate of symmetric best dice using `run_voxel_gnn.py`.


## Spectral shower clustering

Perform spectral clustering with the normalized adjacency matrix and column-pivoted QR factorization on sparse matrices.

The spectral clusterer is implemented in the `pi0_reco` under `pi0/utils/spectral_clusterer.py`, and additional analysis can be found under `/gpfs/slac/staas/fs1/g/neutrino/qatom/spectral-pi0.ipynb`.

## Cone shower clustering

Perform cone clustering with DBSCAN density-based clustering and a PCA to provide initial estimates of cone size and orientation.

The cone clusterer is implemented in the `pi0_reco` under `pi0/utils/cone_clusterer.py`, and additional analysis can be found under `/gpfs/slac/staas/fs1/g/neutrino/qatom/optimize cone clusterer.ipynb`.


## Full pipeline reconstruction of pi0 mass (cone clustering)

Create a mass peak of the neutral pion from Monte Carlo simulated data.

Currently default is implemented for using cone clustering, but it can easily be extended to spectral clustering using the spectral clusterer in the `pi0_reco` and GNN clustering. Run `mass_reco_cone.py` to generate the intermediate data (setting `OUTPUT_DIR` to the appropriate directory). Plots can be made by running `pi0_plotter.py` (it's written for a Jupyter notebook, so I recommend copy-pasting it into one).