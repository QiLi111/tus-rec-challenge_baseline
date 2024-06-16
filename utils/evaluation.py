import os
import torch,h5py
import numpy as np
from utils.network import build_model
from utils.data_process_functions import *
from utils.transform import LabelTransform, PredictionTransform,Transform2Transform,TransformAccumulation
from utils.funs import *



class Evaluation():  

    def __init__(self, opt,device, dset,model_name,data_pairs,saved_folder):
        self.opt = opt
        self.device = device
        self.dset = dset
        self.model_name = model_name
        self.data_pairs = data_pairs.to(self.device)
        self.saved_folder = saved_folder
        self.tform_calib_scale,self.tform_calib_R_T, self.tform_calib = read_calib_matrices(opt.FILENAME_CALIB)
        # image points coordinates in image coordinate system, all pixel points
        self.image_points = reference_image_points(self.dset[0][0].shape[1:],self.dset[0][0].shape[1:]).to(device)

        self.transform_prediction = PredictionTransform(
            opt.PRED_TYPE,
            opt.LABEL_TYPE,
            num_pairs=self.data_pairs.shape[0],
            image_points=self.image_points,
            tform_image_to_tool=self.tform_calib.to(device),
            tform_image_mm_to_tool=self.tform_calib_R_T.to(device),
            tform_image_pixel_to_mm = self.tform_calib_scale.to(device)
            )
        
        self.transform_to_transform = Transform2Transform(
        # transform prediction into 4*4 transformation matrix
            pred_type=opt.PRED_TYPE,
            num_pairs=self.data_pairs.shape[0],
            image_points=self.image_points,
            tform_image_to_tool=self.tform_calib.to(device),
            tform_image_mm_to_tool=self.tform_calib_R_T.to(device),
            tform_image_pixel_to_mm = self.tform_calib_scale.to(device)
            )
        
        self.transform_accumulation = TransformAccumulation(
            image_points=self.image_points,
            tform_image_to_tool=self.tform_calib.to(device),
            tform_image_mm_to_tool=self.tform_calib_R_T.to(device),
            tform_image_pixel_to_image_mm = self.tform_calib_scale.to(device)
            )


        self.pred_dim = type_dim(self.opt.PRED_TYPE, self.image_points.shape[1], self.data_pairs.shape[0])
        self.label_dim = type_dim(self.opt.LABEL_TYPE, self.image_points.shape[1], self.data_pairs.shape[0])
        
        self.model = build_model(
            self.opt,
            in_frames = self.opt.NUM_SAMPLES,
            pred_dim = self.pred_dim,
            ).to(device)
        ## load the model
        self.model.load_state_dict(torch.load(os.path.join(self.opt.SAVE_PATH,'saved_model', self.model_name),map_location=torch.device(self.device)))
        self.model.train(False)

    def calculate_GT_DDF(self, scan_index):
        # calculate DDF of ground truth - label
        frames, tforms, indices, scan_name = self.dset[scan_index]
        frames, tforms = (torch.tensor(t)[None,...].to(self.device) for t in [frames, tforms])
        tforms_inv = torch.linalg.inv(tforms)
        landmark_file = h5py.File(os.path.join(self.opt.LANDMARK_PATH,"landmark_%03d.h5" %indices[0]), 'r')
        landmark = torch.from_numpy(landmark_file[scan_name][()])

        self.labels_global_allpts_DDF, self.labels_global_four = self.cal_label_globle_allpts(frames,tforms,tforms_inv)
        self.labels_global_landmark_DDF = self.cal_label_globle_landmark(tforms,tforms_inv,landmark)
        self.labels_local_allpts_DDF = self.cal_label_local_allpts(tforms,tforms_inv)
        self.labels_local_landmark_DDF = self.cal_label_local_landmark(tforms,tforms_inv,landmark)

    def cal_label_globle_allpts(self,frames,tforms,tforms_inv):
        # global recosntruction on all points
        data_pairs_all = data_pairs_adjacent(frames.shape[1])[1:,:]
        transform_label_global_all = LabelTransform(
            "point",
            pairs=data_pairs_all,
            image_points=self.image_points,
            tform_image_to_tool=self.tform_calib.to(self.device),
            tform_image_mm_to_tool=self.tform_calib_R_T.to(self.device),
            tform_image_pixel_to_mm = self.tform_calib_scale.to(self.device)
            )
        # the second way to generate global reconstruction DDF
        # tforms_each_frame2frame0_gt = transform_label_global_all(tforms, tforms_inv)
        # # coordinates of points in frame 0 coordinate system (mm), global recosntruction  
        # labels_global_allpts = torch.matmul(tforms_each_frame2frame0_gt,torch.matmul(self.tform_calib_scale.to(self.device),self.image_points))[:,:,0:3,...]
        # coordinates of all points in frame 0 coordinate system (mm), global recosntruction  
        labels_global_allpts = torch.squeeze(transform_label_global_all(tforms, tforms_inv))
        # DDF in mm, from current frame to frame 0
        labels_global_allpts_DDF = labels_global_allpts - torch.matmul(self.tform_calib_scale.to(self.device),self.image_points)[0:3,:].expand(labels_global_allpts.shape[0],-1,-1)
        labels_global_allpts_DDF = labels_global_allpts_DDF.cpu().numpy()
        
        # select 4 corner point to plot the trajectory of the scan
        # can use self.image_points to check the correctness, i.e., self.image_points[...,[0,frames.shape[-1]-1,(frames.shape[-2]-1)*frames.shape[-1],-1]]
        #  this should be tensor(
        # [[  1., 640.,   1., 640.],
        # [  1.,   1., 480., 480.],
        # [  0.,   0.,   0.,   0.],
        # [  1.,   1.,   1.,   1.]])
        labels_global_four = labels_global_allpts[...,[0,frames.shape[-1]-1,(frames.shape[-2]-1)*frames.shape[-1],-1]].cpu().numpy()
        # add the first frame
        first_frame_coord_all = torch.matmul(self.tform_calib_scale,self.image_points.cpu())[0:3,:]
        first_frame_coord = first_frame_coord_all.numpy()[...,[0,frames.shape[-1]-1,(frames.shape[-2]-1)*frames.shape[-1],-1]][None,...]
        labels_global_four = np.concatenate((first_frame_coord,labels_global_four),axis = 0)
        
        return labels_global_allpts_DDF, labels_global_four
    
    def cal_label_globle_landmark(self,tforms,tforms_inv,landmark):
        # global recosntruction on landmark points

        # generate required transformations
        data_pairs_landmark = torch.tensor([[0,n0] for n0 in landmark[:,0]])
        transform_label_global_landmark = LabelTransform(
            "transform",
            pairs=data_pairs_landmark,
            image_points=self.image_points,
            tform_image_to_tool=self.tform_calib.to(self.device),
            tform_image_mm_to_tool=self.tform_calib_R_T.to(self.device),
            tform_image_pixel_to_mm = self.tform_calib_scale.to(self.device)
            )
        
        tforms_each_frame2frame0_gt = torch.squeeze(transform_label_global_landmark(tforms, tforms_inv))
        # coordinates of landmark points in frame 0 coordinate system (mm), global recosntruction  
        labels_global_landmark = torch.zeros(3,len(landmark))
        for i in range(len(landmark)):
            pts_coord = torch.cat((landmark[i][1:], torch.FloatTensor([0,1])),axis = 0).to(self.device)
            labels_global_landmark[:,i] = torch.matmul(tforms_each_frame2frame0_gt[i],torch.matmul(self.tform_calib_scale.to(self.device),pts_coord))[0:3]-torch.matmul(self.tform_calib_scale.to(self.device),pts_coord)[0:3]
        
        labels_global_landmark_DDF = labels_global_landmark.cpu().numpy()

        return labels_global_landmark_DDF
    
    def cal_label_local_allpts(self,tforms,tforms_inv):
        # local recosntruction on all points
        # local transformation means the transformation from current frame to the immediate previous frame
        data_pairs_local_all = data_pairs_local(tforms.shape[1]-1)
        transform_label_local_all = LabelTransform(
            "point",
            pairs=data_pairs_local_all,
            image_points=self.image_points,
            tform_image_to_tool=self.tform_calib.to(self.device),
            tform_image_mm_to_tool=self.tform_calib_R_T.to(self.device),
            tform_image_pixel_to_mm = self.tform_calib_scale.to(self.device)
            )
        
        labels_local_allpts = transform_label_local_all(tforms, tforms_inv)
        labels_local_allpts = torch.squeeze(labels_local_allpts)
        labels_local_allpts_DDF = labels_local_allpts-torch.matmul(self.tform_calib_scale.to(self.device),self.image_points)[0:3,:].expand(labels_local_allpts.shape[0],-1,-1)
        labels_local_allpts_DDF = labels_local_allpts_DDF.cpu().numpy()
        
        return labels_local_allpts_DDF
    
    def cal_label_local_landmark(self,tforms,tforms_inv,landmark):
        # generate required transformations
        data_pairs_local_landmark = torch.tensor([[n0-1,n0] for n0 in landmark[:,0]])
        transform_label_local_landmark = LabelTransform(
            "transform",
            pairs=data_pairs_local_landmark,
            image_points=self.image_points,
            tform_image_to_tool=self.tform_calib.to(self.device),
            tform_image_mm_to_tool=self.tform_calib_R_T.to(self.device),
            tform_image_pixel_to_mm = self.tform_calib_scale.to(self.device)
            )
        
        tforms_each_frame_to_prev_frame = torch.squeeze(transform_label_local_landmark(tforms, tforms_inv))
        # coordinates of landmark points in previous frame coordinate system (mm), local recosntruction  
        labels_local_landmark = torch.zeros(3,len(landmark))
        for i in range(len(landmark)):
            pts_coord = torch.cat((landmark[i][1:], torch.FloatTensor([0,1])),axis = 0).to(self.device)
            labels_local_landmark[:,i] = torch.matmul(tforms_each_frame_to_prev_frame[i],torch.matmul(self.tform_calib_scale.to(self.device),pts_coord))[0:3]-torch.matmul(self.tform_calib_scale.to(self.device),pts_coord)[0:3]
        
        labels_local_landmark_DDF = labels_local_landmark.cpu().numpy()
        return labels_local_landmark_DDF

    def generate_pred_values(self, scan_index):
        # calculate DDF of prediction
 
        frames, tforms, indices, scan_name = self.dset[scan_index]
        frames, tforms = (torch.tensor(t)[None,...].to(self.device) for t in [frames, tforms])
        frames = frames/255
        landmark_file = h5py.File(os.path.join(self.opt.LANDMARK_PATH,"landmark_%03d.h5" %indices[0]), 'r')
        landmark = torch.from_numpy(landmark_file[scan_name][()])

        transformation_local, transformation_global, self.predictions_global_allpts_DDF, self.predictions_global_four = self.cal_pred_globle_allpts(frames)
        self.pred_global_landmark_DDF = self.cal_pred_globle_landmark(transformation_global,landmark)
        self.prediction_local_allpts_DDF = self.cal_pred_local_allpts(transformation_local)
        self.pred_local_landmark_DDF = self.cal_pred_local_landmark(transformation_local,landmark)


    def cal_pred_globle_allpts(self,frames):
        # generate global displacement vectors for pixel reconstruction
        # NOTE: when self.opt.NUM_SAMPLES is larger than 2, the coords of the last (self.opt.NUM_SAMPLES-2) frames will not be generated
        predictions_global_allpts = torch.zeros((frames.shape[1]-1,3,self.image_points.shape[-1]))
        # local transformation, i.e., transformation from current frame to the immediate previous frame
        transformation_local = torch.zeros(frames.shape[1]-1,4,4)
        # global transformation, i.e., transformation from current frame to the first frame
        transformation_global = torch.zeros(frames.shape[1]-1,4,4)


        prev_transf = torch.eye(4).to(self.device)
        idx_f0 = 0  # this is the reference frame for network prediction
        Pair_index = 0 # select which pair of prediction to be used
        interval_pred = torch.squeeze(self.data_pairs[Pair_index])[1] - torch.squeeze(self.data_pairs[Pair_index])[0]

        while True:
            
            frames_sub = frames[:,idx_f0:idx_f0 + self.opt.NUM_SAMPLES, ...].to(self.device)
            with torch.no_grad():
                outputs = self.model(frames_sub)
                # transform prediction into transformtion, to be accumulated
                preds_transf = self.transform_to_transform(outputs)[0,Pair_index,...] # use the transformtion from image 1 to 0
                transformation_local[idx_f0] = preds_transf
                pts_cord, prev_transf = self.transform_accumulation(prev_transf,preds_transf)
                transformation_global[idx_f0] = prev_transf
                predictions_global_allpts[idx_f0] = pts_cord[0:3,...].cpu() # global displacement vectors for pixel reconstruction


            idx_f0 += interval_pred
            # NOTE: Due to this break, when self.opt.NUM_SAMPLES is larger than 2, the coords of the last (self.opt.NUM_SAMPLES-2) frames will not be generated
            if (idx_f0 + self.opt.NUM_SAMPLES) > frames.shape[1]:
                break

        if self.opt.NUM_SAMPLES > 2:
            # NOTE As not all frames will be reconstructed, we could use interpolation or some other methods to fill the missing frames
            # This baseline code provides a simple way to fill the missing frames, by using the last reconstructed frame to fill the missing frames
            transformation_local[idx_f0:,...] = torch.eye(4).expand(transformation_local[idx_f0:,...].shape[0],-1,-1)
            transformation_global[idx_f0:,...] = transformation_global[idx_f0-1].expand(transformation_global[idx_f0:,...].shape[0],-1,-1)
            predictions_global_allpts[idx_f0:,...] = predictions_global_allpts[idx_f0-1].expand(predictions_global_allpts[idx_f0:,...].shape[0],-1,-1)


        predictions_global_allpts_DDF = predictions_global_allpts.numpy() -torch.matmul(self.tform_calib_scale.to(self.device),self.image_points)[0:3,:].expand(predictions_global_allpts.shape[0],-1,-1).cpu().numpy()

        # store coordinates of four corner points for plot
        predictions_global_four = predictions_global_allpts[...,[0,frames.shape[-1]-1,(frames.shape[-2]-1)*frames.shape[-1],-1]]
        # add the first frame
        first_frame_coord_all = torch.matmul(self.tform_calib_scale,self.image_points.cpu())[0:3,:]
        first_frame_coord = first_frame_coord_all.numpy()[...,[0,frames.shape[-1]-1,(frames.shape[-2]-1)*frames.shape[-1],-1]][None,...]
        predictions_global_four = np.concatenate((first_frame_coord,predictions_global_four),axis = 0)
        
        return transformation_local, transformation_global,predictions_global_allpts_DDF,predictions_global_four
        
    def cal_pred_globle_landmark(self,transformation_global,landmark):
        # generate global displacement vectors for landmark reconstruction
        # transformation_global: transformation from current frame to the first frame
        # landmark: coordinates of landmark points in image coordinate system (in pixel)

        pred_global_landmark = torch.zeros(3,len(landmark))
        for i in range(len(landmark)):    
            pts_coord = torch.cat((landmark[i][1:], torch.FloatTensor([0,1])),axis = 0).to(self.device)
            pred_global_landmark[:,i] = torch.matmul(transformation_global[landmark[i][0]-1].to(self.device),torch.matmul(self.tform_calib_scale.to(self.device),pts_coord))[0:3]-torch.matmul(self.tform_calib_scale.to(self.device),pts_coord)[0:3]
    
        pred_global_landmark_DDF = pred_global_landmark.cpu().numpy()

        return pred_global_landmark_DDF
    
    def cal_pred_local_allpts(self,transformation_local):
        # generate local displacement vectors for pixel reconstruction
        prediction_local_allpts = torch.matmul(transformation_local.to(self.device),torch.matmul(self.tform_calib_scale.to(self.device),self.image_points))
        prediction_local_allpts_DDF = prediction_local_allpts[:,0:3,:]-torch.matmul(self.tform_calib_scale.to(self.device),self.image_points)[0:3,:].expand(prediction_local_allpts.shape[0],-1,-1)
        prediction_local_allpts_DDF = prediction_local_allpts_DDF.cpu().numpy()
        
        return prediction_local_allpts_DDF

    def cal_pred_local_landmark(self,transformation_local,landmark):
        # generate local displacement vectors for landmark reconstruction

        pred_local_landmark = torch.zeros(3,len(landmark))
        for i in range(len(landmark)):    
            pts_coord = torch.cat((landmark[i][1:], torch.FloatTensor([0,1])),axis = 0).to(self.device)
            pred_local_landmark[:,i] = torch.matmul(transformation_local[landmark[i][0]-1].to(self.device),torch.matmul(self.tform_calib_scale.to(self.device),pts_coord))[0:3]-torch.matmul(self.tform_calib_scale.to(self.device),pts_coord)[0:3]
    
        pred_local_landmark_DDF = pred_local_landmark.cpu().numpy()

        return pred_local_landmark_DDF
   
    def scan_plot(self,scan_index):
        # plot the scan in 3D

        frames, tforms, indices, scan_name = self.dset[scan_index]
        frames, tforms = (torch.tensor(t) for t in [frames, tforms])
        frames = frames/255

        labels_four = self.labels_global_four
        pred_four = self.predictions_global_four
        
        saved_img_path = os.path.join(self.saved_folder, "imgs")
        if not os.path.exists(saved_img_path):
            os.makedirs(saved_img_path)

        color = ['g','r']
        # plot label and prediction separately
        plot_scan(labels_four,frames,os.path.join(saved_img_path,'sub%03d__%s' % (indices[0],scan_name)+'_label'),step = frames.shape[0]-1,color = color[0],width = 4, scatter = 8, legend_size=50, legend = 'GT')
        plot_scan(pred_four,frames,os.path.join(saved_img_path,'sub%03d__%s' % (indices[0],scan_name)+'_pred'),step = frames.shape[0]-1,color = color[1],width = 4, scatter = 8, legend_size=50, legend = 'Pred')
        # plot label and prediction in the same figure 
        plot_scan_label_pred(labels_four,pred_four,frames,color,os.path.join(saved_img_path,'sub%03d__%s' % (indices[0],scan_name)+'_pred_label'),step = frames.shape[0]-1,width = 4, scatter = 8, legend_size=50)
