
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


opt = TrainOptions().parse()
writer = SummaryWriter(os.path.join(opt.SAVE_PATH))
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# get data pairs for prediction
data_pairs = pair_samples(opt.NUM_SAMPLES, opt.NUM_PRED, 0).to(device)
with open(opt.SAVE_PATH +'/'+ 'data_pairs.json', 'w', encoding='utf-8') as fp:
    json.dump(data_pairs.cpu().numpy().tolist(), fp, ensure_ascii=False, indent=4)

# all avaliable data
dataset_all = Dataset(
    data_path=opt.DATA_PATH,
    num_samples=opt.NUM_SAMPLES,
    sample_range=opt.SAMPLE_RANGE
    )

## split the entair dataset into Train, Val, and Test
dset_folds = dataset_all.partition_by_ratio(
    ratios = [1]*5, 
    randomise=True, 
    )
# save the indices of the splited train,val and test dataset, for reproducibility
for (idx, ds) in enumerate(dset_folds):
    ds.write_json(os.path.join(opt.SAVE_PATH,"fold_{:02d}.json".format(idx))) 

# construct the train, val and test dataset
dset_train = dset_folds[0]+dset_folds[1]+dset_folds[2]
dset_val = dset_folds[3]
dset_test = dset_folds[4]

# data loader
train_loader = torch.utils.data.DataLoader(
    dset_train,
    batch_size=opt.MINIBATCH_SIZE,
    shuffle=True,
    num_workers=8,
    pin_memory=True
    )

val_loader = torch.utils.data.DataLoader(
    dset_val,
    batch_size=1, 
    shuffle=False,
    num_workers=8,
    pin_memory=True
    )

# a=0

# for step, (frames,tforms,_) in tqdm(enumerate(train_loader)):
#     frames = frames.to(device)
#     a+=1
# print(a) 

## load calibration metric
tform_calib_scale,tform_calib_R_T, tform_calib = read_calib_matrices(opt.FILENAME_CALIB)
# image points coordinates on image coordinate system
image_points = reference_image_points(dset_train[0][0].shape[1:],2).to(device)
# hyper-parameter for prediction and label
pred_dim = type_dim(opt.PRED_TYPE, image_points.shape[1], data_pairs.shape[0])
label_dim = type_dim(opt.LABEL_TYPE, image_points.shape[1], data_pairs.shape[0])
# transform label and prediction into another format, e.g., transform transformation matrix into points
transform_label = LabelTransform(
    opt.LABEL_TYPE,
    pairs=data_pairs,
    image_points=image_points,
    tform_image_to_tool=tform_calib.to(device),
    tform_image_mm_to_tool=tform_calib_R_T.to(device)
    )

transform_prediction = PredictionTransform(
    opt.PRED_TYPE,
    opt.LABEL_TYPE,
    num_pairs=data_pairs.shape[0],
    image_points=image_points,
    tform_image_to_tool=tform_calib.to(device),
    tform_image_mm_to_tool=tform_calib_R_T.to(device)
    )
transform_into_points = PointTransform(
    label_type=opt.LABEL_TYPE,
    image_points=image_points,
    tform_image_to_tool=tform_calib.to(device),
    tform_image_mm_to_tool=tform_calib_R_T.to(device)
    )

## network
model = build_model(
    opt,
    in_frames = opt.NUM_SAMPLES,
    pred_dim = pred_dim,
    ).to(device)

# retrain the model from previous epoch
if opt.retrain:
    model.load_state_dict(torch.load(os.path.join(opt.SAVE_PATH,'saved_model', 'model_epoch'+str(opt.retrain_epoch)),map_location=torch.device(device)))


if opt.multi_gpu:
    model= nn.DataParallel(model)
    print('multi-gpu')
    print(os.environ["CUDA_VISIBLE_DEVICES"])

## training
val_loss_min = 1e10
val_dist_min = 1e10
optimiser = torch.optim.Adam(model.parameters(), lr=opt.LEARNING_RATE)
criterion = torch.nn.MSELoss()
metrics = PointDistance()
print('Training started')
for epoch in range(int(opt.retrain_epoch), int(opt.retrain_epoch)+opt.NUM_EPOCHS):
    
    train_epoch_loss = 0
    train_epoch_dist = 0
    for step, (frames, tforms,_) in enumerate(train_loader):
        frames, tforms = frames.to(device), tforms.to(device)
        tforms_inv = torch.linalg.inv(tforms)
        frames = frames/255
        # transform label based on label type
        labels = transform_label(tforms, tforms_inv)

        optimiser.zero_grad()
        # model prediction
        outputs = model(frames)
        # transform prediction according to label type
        preds = transform_prediction(outputs)
        # calculate loss and metric
        loss = criterion(preds, labels)
        loss.backward()
        optimiser.step()

        # transfrom prediction and label into points, for metric calculation
        preds_pts = transform_into_points(preds.data)
        labels_pts = transform_into_points(labels)
        dist = metrics(preds_pts, labels_pts).detach()
    
        train_epoch_loss += loss.item()
        train_epoch_dist += dist
        
    train_epoch_loss /= (step + 1)
    train_epoch_dist /= (step + 1)
    # print loss information on terminal
    print_info(epoch,train_epoch_loss,train_epoch_dist,opt,'train')

    # validation    
    if epoch in range(int(opt.retrain_epoch), int(opt.retrain_epoch)+opt.NUM_EPOCHS, opt.val_fre):

        model.train(False)

        epoch_loss_val = 0
        epoch_dist_val = 0
        for step, (fr_val, tf_val, _) in enumerate(val_loader):

            fr_val, tf_val = fr_val.to(device), tf_val.to(device)
            tf_val_inv = torch.linalg.inv(tf_val)
            # transform label based on label type
            la_val = transform_label(tf_val, tf_val_inv)
            fr_val = fr_val/255

            out_val = model(fr_val)
            # transform prediction
            pr_val = transform_prediction(out_val)
            # calculate loss and metric
            loss_val = criterion(pr_val, la_val)
            pr_val_pts = transform_into_points(pr_val)
            la_val_pts = transform_into_points(la_val)
            dist_val = metrics(pr_val_pts, la_val_pts).detach()

            epoch_loss_val += loss_val.item()
            epoch_dist_val += dist_val


        epoch_loss_val /= (step+1)
        epoch_dist_val /= (step+1)

        # print loss information on terminal
        print_info(epoch,epoch_loss_val,epoch_dist_val,opt,'val')
        # save model at current epoch
        save_model(model,epoch,opt)
        # save best validation model
        val_loss_min, val_dist_min = save_best_network(opt, model, epoch, epoch_loss_val, epoch_dist_val.mean(), val_loss_min, val_dist_min)
        # add to tensorboard
        loss_dists = {'train_epoch_loss': train_epoch_loss, 'train_epoch_dist': train_epoch_dist,'val_epoch_loss':epoch_loss_val,'val_epoch_dist':epoch_dist_val}
        add_scalars(writer, epoch, loss_dists)
        write_to_txt(opt, epoch, loss_dists)

        model.train(True)
        
        