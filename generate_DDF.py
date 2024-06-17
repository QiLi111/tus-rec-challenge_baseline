
# This script is an exmaple to show how to generate the required 4 kinds of DDF.

import os
import torch
import json
from utils.loader import Dataset
from utils.plot_functions import *
from options.train_options import TrainOptions
from utils.funs import *
from utils.evaluation import Evaluation

opt = TrainOptions().parse()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load previously generated data pairs
with open(os.getcwd()+ '/' + opt.SAVE_PATH + '/' +"data_pairs.json", "r", encoding='utf-8') as f:
    data_pairs= json.load(f)
data_pairs=torch.tensor(data_pairs)

# create the test set loader
opt.FILENAME_TEST=opt.FILENAME_TEST+'.json'
dset_test = Dataset.read_json(os.path.join(os.getcwd(),opt.SAVE_PATH,opt.FILENAME_TEST),num_samples = -1)

# generate the folder to save the results
saved_folder_test = os.path.join(os.getcwd(),opt.SAVE_PATH, 'testing','testing_test_results')
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


