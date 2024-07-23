# Functions used during label DDF generation

import os
import torch
import numpy as np
from utils.transform import LabelTransform
from utils.plot_functions import reference_image_points,read_calib_matrices,data_pairs_adjacent,data_pairs_local

class Transf2DDFs():  

    def __init__(self,data_path_calib):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tform_calib_scale,self.tform_calib_R_T, self.tform_calib = read_calib_matrices(os.path.join(os.path.dirname(os.path.realpath(__file__)),data_path_calib))
        # image points coordinates in image coordinate system, all pixel points
        self.image_points = reference_image_points([480, 640],[480, 640]).to(self.device)

    def calculate_GT_DDF(self, frames,tforms,landmark):
        # calculate DDF of ground truth - label
        frames, tforms = (torch.tensor(t)[None,...].to(self.device) for t in [frames, tforms])
        tforms_inv = torch.linalg.inv(tforms)
        landmark = torch.from_numpy(landmark)
        # Global displacement vectors for pixel reconstruction
        labels_global_allpts_DDF, self.labels_global_four = self.cal_label_globle_allpts(frames,tforms,tforms_inv)
        # Global displacement vectors for landmark reconstruction
        labels_global_landmark_DDF = self.cal_label_globle_landmark(tforms,tforms_inv,landmark)
        # Local displacement vectors for pixel reconstruction
        labels_local_allpts_DDF = self.cal_label_local_allpts(tforms,tforms_inv)
        # Local displacement vectors for landmark reconstruction
        labels_local_landmark_DDF = self.cal_label_local_landmark(tforms,tforms_inv,landmark)

        return labels_global_allpts_DDF,labels_global_landmark_DDF,labels_local_allpts_DDF,labels_local_landmark_DDF


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
    
        # coordinates of all points in frame 0 coordinate system (mm), global recosntruction  
        labels_global_allpts = torch.squeeze(transform_label_global_all(tforms, tforms_inv))
        # DDF in mm, from current frame to frame 0
        labels_global_allpts_DDF = labels_global_allpts - torch.matmul(self.tform_calib_scale.to(self.device),self.image_points)[0:3,:].expand(labels_global_allpts.shape[0],-1,-1)
        labels_global_allpts_DDF = labels_global_allpts_DDF.cpu().numpy()
        
        # select 4 corner points to plot the trajectory of the scan
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
        # transformation from the current frame to frame 0
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
        
        # coordinates of all points in previous frame coordinate system (mm), local recosntruction  
        labels_local_allpts = transform_label_local_all(tforms, tforms_inv)
        labels_local_allpts = torch.squeeze(labels_local_allpts)
        # calculate DDF, displacement from current frame to the immediate previous frame
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
        # transformation from current frame to previous frame
        tforms_each_frame_to_prev_frame = torch.squeeze(transform_label_local_landmark(tforms, tforms_inv))
        # coordinates of landmark points in previous frame coordinate system (mm), local recosntruction  
        labels_local_landmark = torch.zeros(3,len(landmark))
        for i in range(len(landmark)):
            pts_coord = torch.cat((landmark[i][1:], torch.FloatTensor([0,1])),axis = 0).to(self.device)
            labels_local_landmark[:,i] = torch.matmul(tforms_each_frame_to_prev_frame[i],torch.matmul(self.tform_calib_scale.to(self.device),pts_coord))[0:3]-torch.matmul(self.tform_calib_scale.to(self.device),pts_coord)[0:3]
        
        labels_local_landmark_DDF = labels_local_landmark.cpu().numpy()
        return labels_local_landmark_DDF

