# =========== OUTPUT REQUIRMENT ============
# please see the following requirment from Challenge task, to generate the required data for final evaluation and rank

# When testing, the algorithm is expected to take the entire scan as input and output two different sets of transformation-representing displacement vectors as results, a set of displacement vectors on individual pixels and a set of displacement vectors on provided landmarks. 
# There is no requirement on how the algorithm is designed internally, for example, whether it is learning-based method; frame-, sequence- or scan-based processing; or, rigid-, affine- or nonrigid transformation assumptions.

# This script is an exmaple to show the evaluation process during FINAL testing, where only "frames" in each scan is accessible, and the "tforms" is not provided.
# Each folder includes 24 .h5 files, each denoting one scan. In each scan, only "frames" are accessible, denoting all the frames in that scan
# In order to show clearly the structure of the test dataset, the predicted .h5 file is pre-generated and provided, with keys only.
# The format of keys is "sub%03d_%s" where %03d denotes the subfolder index, and %s denotes the scan name.
# The paticipants can index the coorsponding .h5 file based on the keys in prediction.h5
# All the labels (in the format of coordinates) are stored in four .h5 files, for calculating error
# This script also provide the label generation process. 

import os
import torch
import json
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from utils.loader import Dataset
from utils.network import build_model
from utils.loss import PointDistance
from utils.data_process_functions import *
from utils.transform import LabelTransform, PredictionTransform, PointTransform
from options.train_options import TrainOptions
from utils.funs import *
from utils.evaluation import Evaluation


opt = TrainOptions().parse()
writer = SummaryWriter(os.path.join(opt.SAVE_PATH))
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

saved_folder_train = os.path.join(opt.SAVE_PATH, 'testing', 'testing_train_results')
saved_folder_val = os.path.join(opt.SAVE_PATH,'testing', 'testing_val_results')
saved_folder_test = os.path.join(opt.SAVE_PATH, 'testing','testing_test_results')
if not os.path.exists(saved_folder_train):
    os.makedirs(saved_folder_train)
if not os.path.exists(saved_folder_val):
    os.makedirs(saved_folder_val)
if not os.path.exists(saved_folder_test):
    os.makedirs(saved_folder_test)

# During testing, the data path should be mounted to a folder where the testing data is stored
opt.LABEL_PATH = saved_folder_test # For develop use
opt.PREDICTION_PATH = saved_folder_test
# evaluation on training set
Evaluation_test = Evaluation(opt,device, dset_test,'best_validation_dist_model',data_pairs)
# generate labels and predictions keys for each scan - by Challenge Orgnisors (for reference use only)
# to simulate real testing
# for scan_index in range(len(dset_test)):
#     # generate keys in prediction .h5 file 
#     Evaluation_test.generate_pred_keys(scan_index)
#     # get gruond truth labels 
#     Evaluation_test.calculate_GT_DDF(scan_index) 
    
# # # close all .h5 files
Evaluation_test.close_pred_keys_files()
# Evaluation_test.close_label_files()

# The following is what the participants need to do - generating four DDF
# load predifined prediction .h5 
Evaluation_test.load_local_h5()
# generate 4 kinds of predictions, based on the provided keys
Evaluation_test.generate_prediction_h5()
for scan_index in range(len(list(Evaluation_test.Prediction_Global_AllPts_keys_only.keys()))):
    # predictions
    Evaluation_test.calculate_GT_DDF(scan_index)
    Evaluation_test.generate_pred_values(scan_index)
    Evaluation_test.scan_plot(scan_index)


# evaluation on validation set
opt.LABEL_PATH = saved_folder_val
opt.PREDICTION_PATH = saved_folder_val
Evaluation_val = Evaluation(opt,device, dset_val,'best_validation_dist_model',data_pairs)
# evaluation on test set
opt.LABEL_PATH = saved_folder_test
opt.PREDICTION_PATH = saved_folder_test
Evaluation_test = Evaluation(opt,device, dset_test,'best_validation_dist_model',data_pairs)


