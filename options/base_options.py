import argparse
import os

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--DATA_PATH', type=str, default='/raid/Qi/public_data/forearm_US_large_dataset/TrainData_50_MICCAIChallenge/DataSet', help='foldername of dataset path')
        self.parser.add_argument('--FILENAME_CALIB', type=str, default="/raid/Qi/public_data/forearm_US_large_dataset/TrainData_50_MICCAIChallenge/DataSet/calib_matrix.csv",help='dataroot of calibration matrix')
        self.parser.add_argument('--LABEL_PATH', type=str, default='/raid/Qi/public_data/forearm_US_large_dataset/TrainData_50_MICCAIChallenge/DataSet', help='foldername of label path for testing set')
        self.parser.add_argument('--PREDICTION_PATH', type=str, default='/raid/Qi/public_data/forearm_US_large_dataset/TrainData_50_MICCAIChallenge/DataSet/Baseline', help='foldername of prediction path for testing set')
        self.parser.add_argument('--LANDMARK_PATH', type=str, default='/raid/Qi/public_data/forearm_US_large_dataset/TrainData_50_MICCAIChallenge/landmark', help='foldername of label path for testing set')
        self.parser.add_argument('--multi_gpu', type=bool, default=False, help='whether use multi gpus')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu id: e.g., 0,1,2...')
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain

        args = vars(self.opt)

        print('----------Option----------')
        for k, v in sorted(args.items()):
            print('%s, %s' % (str(k), str(v)))
            print('\n')
        print('----------Option----------')

        # create saved result path
        saved_results = 'seq_len' + str(self.opt.NUM_SAMPLES) + '__' + self.opt.model_name + '__' + 'lr' + str(self.opt.LEARNING_RATE)\
        + '__pred_type_'+str(self.opt.PRED_TYPE) + '__label_type_'+str(self.opt.LABEL_TYPE) 
        self.opt.SAVE_PATH = os.path.join(os.getcwd(),'results', saved_results)
        
        if not os.path.exists(self.opt.SAVE_PATH):
            os.makedirs(self.opt.SAVE_PATH)
        if not os.path.exists(os.path.join(self.opt.SAVE_PATH, 'saved_model')):
            os.makedirs(os.path.join(self.opt.SAVE_PATH, 'saved_model'))
        if not os.path.exists(os.path.join(self.opt.SAVE_PATH, 'train_results')):
            os.makedirs(os.path.join(self.opt.SAVE_PATH, 'train_results'))
        if not os.path.exists(os.path.join(self.opt.SAVE_PATH, 'val_results')):
            os.makedirs(os.path.join(self.opt.SAVE_PATH, 'val_results'))

        file_name = os.path.join(self.opt.SAVE_PATH, 'config.txt')
        with open(file_name, 'a') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s,%s' % (str(k), str(v)))
                opt_file.write('\n')
            opt_file.write('------------ Options -------------\n')
        return self.opt
