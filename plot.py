# An example of load data and plot scan
# TUS-REC Challenge
# Trackerless 3D Freehand Ultrasound Reconstruction (TUS-REC) Challenge

import os
import h5py
import torch
from options.train_options import TrainOptions
from utils.plot_functions import *


opt = TrainOptions().parse()
DATA_DIR = opt.DATA_PATH
FILENAME_CALIB = opt.FILENAME_CALIB

# get object names in HDF5 file
DataSet = h5py.File(os.path.join(os.getcwd(),DATA_DIR,"005","RH_Per_S_PtD.h5"),'r')    
for key in DataSet.keys():
    print(key) 
    print(type(DataSet[key])) 

# get all frames in the fifth scan of subject 10 (index from 0)
example_scan = DataSet["frames"][()]
# get associated transformations of all frames in the fifth scan of subject 10, from tracker tool space to camera space
example_transformation = DataSet["tforms"][()]

DataSet.close()

# plot scan in 3D 

# four corner points defined in image coordinate system
points_four = reference_image_points((example_scan.shape[1],example_scan.shape[2]), 2)
# transformations from the current frame to the first frame
data_pairs = data_pairs_adjacent(example_scan.shape[0])
tforms_each_frame2frame0 = transform_t2t(torch.from_numpy(example_transformation), torch.linalg.inv(torch.from_numpy(example_transformation)),data_pairs)
# get calibration matrix, including rigid tarnsformation and scale
tform_calib_scale, tform_calib_R_T, tform_calib = read_calib_matrices(os.path.join(os.getcwd(),FILENAME_CALIB))
# get coordinates of four corner points in 3D space, unit mm
labels_four = torch.matmul(torch.linalg.inv(tform_calib_R_T),torch.matmul(tforms_each_frame2frame0,torch.matmul(tform_calib,points_four)))[:,0:3,...]
# plot scan trajectory 
plot_scan(labels_four.numpy(),example_scan,os.getcwd()+'/'+"example",step=10,color = 'g',width = 4, scatter = 8, legend_size=50,legend = 'GT')

