#!/bin/bash

python extract_hdf5_to_npy.py --root-dir='/local/a/akosta/Datasets/MVSEC/'  --env='indoor_flying1' --sub-dir='evf' --mp
python extract_hdf5_to_npy.py --root-dir='/local/a/akosta/Datasets/MVSEC/'  --env='indoor_flying2' --sub-dir='evf' --mp
python extract_hdf5_to_npy.py --root-dir='/local/a/akosta/Datasets/MVSEC/'  --env='indoor_flying3' --sub-dir='evf' --mp


python extract_hdf5_to_npy.py --root-dir='/local/a/akosta/Datasets/MVSEC/'  --env='outdoor_day1' --sub-dir='evf' --mp
python extract_hdf5_to_npy.py --root-dir='/local/a/akosta/Datasets/MVSEC/'  --env='outdoor_day2' --sub-dir='evf' --mp
