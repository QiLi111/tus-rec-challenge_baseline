
# This script is an exmaple to show how to generate the required 4 kinds of DDF.

import os
import torch
import json
from utils.loader import Dataset
from utils.data_process_functions import *
from options.train_options import TrainOptions
from utils.funs import *
from utils.evaluation import Evaluation

opt = TrainOptions().parse()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load previously generated data pairs
with open(opt.SAVE_PATH + '/' +"data_pairs.json", "r", encoding='utf-8') as f:
    data_pairs= json.load(f)
data_pairs=torch.tensor(data_pairs)

# create the validation/test set loader
opt.FILENAME_VAL=opt.FILENAME_VAL+'.json'
opt.FILENAME_TEST=opt.FILENAME_TEST+'.json'
opt.FILENAME_TRAIN=[opt.FILENAME_TRAIN[i]+'.json' for i in range(len(opt.FILENAME_TRAIN))]

dset_val = Dataset.read_json(os.path.join(opt.SAVE_PATH,opt.FILENAME_VAL),num_samples = -1)
dset_test = Dataset.read_json(os.path.join(opt.SAVE_PATH,opt.FILENAME_TEST),num_samples = -1)
dset_train_list = [Dataset.read_json(os.path.join(opt.SAVE_PATH,opt.FILENAME_TRAIN[i]),num_samples = -1) for i in range(len(opt.FILENAME_TRAIN))]
dset_train = dset_train_list[0]+dset_train_list[1]+dset_train_list[2]

# generate the folder to save the results
saved_folder_train = os.path.join(opt.SAVE_PATH, 'testing', 'testing_train_results')
saved_folder_val = os.path.join(opt.SAVE_PATH,'testing', 'testing_val_results')
saved_folder_test = os.path.join(opt.SAVE_PATH, 'testing','testing_test_results')
if not os.path.exists(saved_folder_train):
    os.makedirs(saved_folder_train)
if not os.path.exists(saved_folder_val):
    os.makedirs(saved_folder_val)
if not os.path.exists(saved_folder_test):
    os.makedirs(saved_folder_test)

# generate DDF on test set
Evaluation_test = Evaluation(opt,device, dset_test,'best_validation_dist_model',data_pairs,saved_folder_test)
for scan_index in range(len(dset_test)):
    # generate 4 DDFs for each scan, using labels from the dataset
    Evaluation_test.calculate_GT_DDF(scan_index) 
    # generate 4 DDFs for each scan, using prediction from the model
    Evaluation_test.generate_pred_values(scan_index)
    # plot the scan
    Evaluation_test.scan_plot(scan_index)
    print('%4d scans is finished.'%scan_index)

    
# generate DDF on val set
Evaluation_val = Evaluation(opt,device, dset_val,'best_validation_dist_model',data_pairs,saved_folder_val)
for scan_index in range(len(dset_val)):
    # generate 4 DDFs for each scan, using labels from the dataset
    Evaluation_val.calculate_GT_DDF(scan_index) 
    # generate 4 DDFs for each scan, using prediction from the model
    Evaluation_val.generate_pred_values(scan_index)
    # plot the scan
    Evaluation_val.scan_plot(scan_index)
    print('%4d scans is finished.'%scan_index)

# generate DDF on train set
Evaluation_train = Evaluation(opt,device, dset_train,'best_validation_dist_model',data_pairs,saved_folder_train)
for scan_index in range(len(dset_train)):
    # generate 4 DDFs for each scan, using labels from the dataset
    Evaluation_train.calculate_GT_DDF(scan_index) 
    # generate 4 DDFs for each scan, using prediction from the model
    Evaluation_train.generate_pred_values(scan_index)
    # plot the scan
    Evaluation_train.scan_plot(scan_index)
    print('%4d scans is finished.'%scan_index)



