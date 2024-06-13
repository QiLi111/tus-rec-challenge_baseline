from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        self.parser.add_argument('--PRED_TYPE', type=str,default='parameter',help='network output type: {"transform", "parameter", "point"}')
        self.parser.add_argument('--LABEL_TYPE', type=str,default='point',help='label type: {"point", "parameter"}')
        self.parser.add_argument('--NUM_SAMPLES', type=int,default=10,help='number of input frames/imgs')
        self.parser.add_argument('--SAMPLE_RANGE', type=int,default=10,help='from which the input frames/imgs are selected from')
        self.parser.add_argument('--NUM_PRED', type=int,default=9,help='to those frames/imgs, transformation matrix are predicted ')
        self.parser.add_argument('--model_name', type=str,default='efficientnet_b1',help='network name:{"efficientnet_b1", "resnet", "LSTM_0", "LSTM", "LSTM_GT","classicifation_b1"}')

        self.parser.add_argument('--retrain', type=bool,default=False,help='whether load a pretrained model')
        self.parser.add_argument('--retrain_epoch', type=str,default='00000000',help='whether load a pretrained model: {0: train from sctrach; a number, e.g., 1000, train from epoch 1000}')
        self.parser.add_argument('--MINIBATCH_SIZE', type=int,default=4,help='input batch size')
        self.parser.add_argument('--LEARNING_RATE',type=float,default=1e-4,help='learing rate')
        self.parser.add_argument('--NUM_EPOCHS',type =int,default=int(1e6),help='# of iter to lin')
        self.parser.add_argument('--FREQ_INFO', type=int, default=10,help='frequency of print info')
        self.parser.add_argument('--FREQ_SAVE', type=int, default=100,help='frequency of save model')
        self.parser.add_argument('--val_fre', type=int, default=1,help='frequency of validation')

        #######used in testing##################################################
        self.parser.add_argument('--FILENAME_VAL', type=str, default="fold_03", help='validation json file')
        self.parser.add_argument('--FILENAME_TEST', type=str, default="fold_04", help='test json file')
        self.parser.add_argument('--FILENAME_TRAIN', type=list,default=["fold_00", "fold_01", "fold_02"],help='train json file')
        self.parser.add_argument('--MODEL_FN', type=str, default="saved_model/", help='model path for visulize')

        self.isTrain= True
