# Transformation functions
import torch
import pytorch3d.transforms
from utils.rigid_transform_3D import rigid_transform_3D

''' Geometric transformation used in transforms:
Use left-multiplication:
    T{image->world} = T{tool->world} * T{image->tool} 
    T{image->tool} is the calibration matrix: T_calib
    pts{tool} = T{image->tool} * pts{image}
    pts{world} = T{image->world} * pts{image}
        where pts{image1} are four (corners) or five (and centre) image points in the image coordinate system.

The ImagePointTransformLoss 
    - a pair of "past" and "pred" images, image0 and image1
    - a pair of ground-truth (GT) and predicted (Pred) transformation
    pts{world} = T{tool0->world} * T_calib * pts{image0}
    pts{world} = T{tool1->world} * T_calib * pts{image1}
    => T{tool0->world} * T_calib * pts{image0} = T{tool1->world} * T_calib * pts{image1}
    => pts{image0} = T_calib^(-1) * T{tool0->world}^(-1) * T{tool1->world} * T_calib * pts{image1}
    => pts{image0} = T_calib^(-1) * T(tool1->tool0) * T_calib * pts{image1}

Denote world-coordinates-independent transformation, 
    T(tool1->tool0) = T{tool0->world}^(-1) * T{tool1->world}, 
    which can be predicted and its GT can be obtained from the tracker.
When image_coords = True (pixel): 
    => loss = ||pts{image0}^{GT} - pts{image0}^{Pred}||
When image_coords = False (mm):
    => loss = ||pts{tool0}^{GT} - pts{tool0}^{Pred}||
        where pts{tool0} = T(tool1->tool0) * T_calib * pts{image1}

Accumulating transformations using TransformAccumulation class:
    Given often-predicted T(tool1->tool0) and then T(tool2->tool1):
        T(tool2->tool0) = T(tool1->tool0) * T(tool2->tool1)
    Similarly then: 
        pts{image0} = T_calib^(-1) * T(tool2->tool0) * T_calib * pts{image2}

'''


class LabelTransform():

    def __init__(
        self, 
        label_type, 
        pairs,
        image_points=None,
        tform_image_to_tool=None,
        tform_image_mm_to_tool = None,
        tform_image_pixel_to_mm = None
        ):        
        """
        :param label_type: {"point", "parameter"}
        :param pairs: data pairs, between which the transformations are constructed, returned from pair_samples
        :param image_points: 2-by-num or 3-by-num in pixel, used for computing loss
        :param tform_image_to_tool: transformation from image coordinates to tool coordinates, usually obtained from calibration
        :param tform_image_mm_to_tool: transformation from image coordinates (mm) to tool coordinates
        :param tform_image_pixel_to_mm: transformation from image coordinates (pixel) to image coordinates (mm)
        """

        self.label_type = label_type
        self.pairs = pairs
        self.tform_image_to_tool = tform_image_to_tool
        self.tform_image_mm_to_tool = tform_image_mm_to_tool
        self.tform_image_pixel_to_mm = tform_image_pixel_to_mm
        self.image_points = image_points
        # pre-compute reference points in tool coordinates
        self.image_points_in_tool = torch.matmul(self.tform_image_to_tool,self.image_points)
        self.tform_tool_to_image = torch.linalg.inv(self.tform_image_to_tool)
        self.tform_tool_to_image_mm = torch.linalg.inv(self.tform_image_mm_to_tool)
        self.tform_image_mm_to_pixel = torch.linalg.inv(self.tform_image_pixel_to_mm) 

        if self.label_type=="point":              
            self.call_function = self.to_points
        
        elif self.label_type=="transform":  
            self.call_function = self.to_transform_t2t   

        elif self.label_type=="parameter":
            self.call_function = self.to_parameters   

        else:
            raise('Unknown label_type!')
    

    def __call__(self, *args, **kwargs):
        return self.call_function(*args, **kwargs)        
    
    def to_points(self, tforms, tforms_inv=None):
        _tforms = self.to_transform_t2t(tforms, tforms_inv)
        return torch.matmul(_tforms, torch.matmul(self.tform_image_pixel_to_mm,self.image_points))[:,:,0:3,:]  # [batch,num_pairs,(x,y,z,1),num_image_points]
 
    def to_transform_t2t(self, tforms, tforms_inv):
        # the label includes the rigid part of calibration matrix, so the transformation is from image(mm) to image(mm), and the final transformed points is in mm, rather in pixel
        # such that the label is Orthogonal Matrix, and then the label could be converted to 6DOF parameter using functions in pytorch3d
        if tforms_inv is None:
            tforms_inv = torch.linalg.inv(tforms)
        
        tforms_world_to_tool0 = tforms_inv[:,self.pairs[:,0],:,:]
        tforms_tool1_to_world = tforms[:,self.pairs[:,1],:,:]
        # tform_tool1_to_tool0 is Orthogonal Matrix
        tform_tool1_to_tool0 = torch.matmul(tforms_world_to_tool0, tforms_tool1_to_world)  # tform_tool1_to_tool0
        # the returned matrix, which is the label, is Orthogonal Matrix
        return torch.matmul(self.tform_tool_to_image_mm[None,None,...],torch.matmul(tform_tool1_to_tool0,self.tform_image_mm_to_tool[None,None,...])) # tform_image1_mm_to_image0_mm

    def to_parameters(self, tforms, tforms_inv):
        _tforms = self.to_transform_t2t(tforms, tforms_inv)
        # only Orthogonal Matrix can be converted to Euler angles
        Rotation = pytorch3d.transforms.matrix_to_euler_angles(_tforms[:,:,0:3, 0:3], 'ZYX')
        params = torch.cat((Rotation, _tforms[:,:,0:3, 3]),2)
        return params


class PredictionTransform():

    def __init__(
        self, 
        pred_type, 
        label_type, 
        num_pairs=None,
        image_points=None,
        tform_image_to_tool=None,
        tform_image_mm_to_tool=None,
        tform_image_pixel_to_mm = None
        ):
        
        """
        :param pred_type = {"transform", "parameter", "point"}
        :param label_type = {"point", "parameter"}
        :param pairs: data pairs, between which the transformations are constructed, returned from pair_samples
        :param image_points: 2-by-num or 3-by-num in pixel, used for computing loss
        :param tform_image_to_tool: transformation from image coordinates to tool coordinates, usually obtained from calibration
        :param tform_image_mm_to_tool: transformation from image coordinates (mm) to tool coordinates
        :param tform_image_pixel_to_mm: transformation from image coordinates (pixel) to image coordinates (mm)
        """

        self.pred_type = pred_type
        self.label_type = label_type
        self.num_pairs = num_pairs

        self.image_points = image_points
        self.tform_image_to_tool = tform_image_to_tool
        self.tform_image_mm_to_tool = tform_image_mm_to_tool
        self.tform_image_pixel_to_mm = tform_image_pixel_to_mm
        self.image_points_in_tool = torch.matmul(self.tform_image_to_tool,self.image_points)
        self.tform_tool_to_image = torch.linalg.inv(self.tform_image_to_tool)
        self.tform_tool_to_image_mm = torch.linalg.inv(self.tform_image_mm_to_tool)
        self.tform_image_mm_to_pixel = torch.linalg.inv(self.tform_image_pixel_to_mm) 


        if self.pred_type=="point":
            if self.label_type=="point":
                self.call_function = self.point_to_point
            elif self.label_type=="parameter":
                raise("Not supported!")
            elif self.label_type=="transform":
                raise("Not supported!")
            else:
                raise('Unknown label_type!')
            
        else:                            
            if self.pred_type=="parameter":
                if self.label_type=="point":
                    self.call_function = self.parameter_to_point
                elif self.label_type=="parameter":
                    self.call_function = self.parameter_to_parameter
                elif self.label_type == "transform":
                    self.call_function = self.param_to_transform
                else:
                    raise('Unknown label_type!')

            elif self.pred_type=="transform":
                if self.label_type=="point":
                    self.call_function = self.transform_to_point
                elif self.label_type=="parameter":
                # as the prediction is not constrained to be Orthogonal Matrix, the transformation cannot converted into 6DOF parameter using functions in pytorch3d
                    raise('Not supported!')
                elif self.label_type == "transform":
                    self.call_function = self.transform_to_transform

                else:
                    raise('Unknown label_type!')
            
            elif self.pred_type=="quaternion":
                if self.label_type=="point":
                    self.call_function = self.quaternion_to_point
                elif self.label_type=="parameter":
                    self.call_function = self.quaternion_to_parameter
                elif self.label_type == "transform":
                    self.call_function = self.quaternion_to_transform

                else:
                    raise('Unknown label_type!')

            else:
                raise('Unknown pred_type!')
        

    def __call__(self, outputs):
        preds = outputs.reshape((outputs.shape[0],self.num_pairs,-1))
        return self.call_function(preds)

    def transform_to_parameter(self, _tforms):
        last_rows = torch.cat([
            torch.zeros_like(_tforms[..., 0])[..., None, None].expand(-1, -1, 1, 3),
            torch.ones_like(_tforms[..., 0])[..., None, None]
        ], axis=3)
        _tforms = torch.cat((
            _tforms.reshape(-1, self.num_pairs, 3, 4),
            last_rows
        ), axis=2)
        Rotation = pytorch3d.transforms.matrix_to_euler_angles(_tforms[:, :, 0:3, 0:3], 'ZYX')
        params = torch.cat((Rotation, _tforms[:, :, 0:3, 3]),2)
        return params

    def point_to_parameter(self, pts):
        # rigid_transform_3D(A, B) --> transformation from A to B
        pts = pts.reshape(pts.shape[0],self.num_pairs,3,-1)
        last_rows = torch.unsqueeze(torch.ones_like(pts[:,:,0,:]),2)
        transf = torch.ones_like(pts)
        pts = torch.cat((pts,last_rows), axis=2)
        
        
        for i in range(pts.size()[0]):
            for j in range(pts.size()[1]):
                ret_R, ret_t = rigid_transform_3D(torch.matmul(self.tform_image_pixel_to_mm,self.image_points)[0:3,:],pts[i,j,0:3,:])
                transf[i,j,...] = torch.cat((ret_R,ret_t),1)
        # ret_R, ret_t = rigid_transform_3D(self.image_points_in_tool[0:3,:].repeat(pts.size()[0],pts.size()[1],1,1),pts[:,:,0:3,:])
        
        Rotation = pytorch3d.transforms.matrix_to_euler_angles(transf[:,:,0:3, 0:3], 'ZYX')
        params = torch.cat((Rotation, transf[:,:,0:3, 3]),2)
        return params
        

    def point_to_point(self,pts):
        return pts.reshape(pts.shape[0],self.num_pairs,3,-1)


    def transform_to_transform(self, transform):
        last_rows = torch.cat([
            torch.zeros_like(transform[..., 0])[..., None, None].expand(-1, -1, 1, 3),
            torch.ones_like(transform[..., 0])[..., None, None]
            ], axis=3)
        transform = torch.cat((
            transform.reshape(-1, self.num_pairs, 3, 4),
            last_rows
            ), axis=2)

        return transform

    def transform_to_point(self,_tforms):
        
        last_rows = torch.cat([
            torch.zeros_like(_tforms[...,0])[...,None,None].expand(-1,-1,1,3),
            torch.ones_like(_tforms[...,0])[...,None,None]
            ], axis=3)
        _tforms = torch.cat((
            _tforms.reshape(-1,self.num_pairs,3,4),
            last_rows
            ), axis=2)
        
        return torch.matmul(_tforms, torch.matmul(self.tform_image_pixel_to_mm, self.image_points))[:,:,0:3,:]  # [batch,num_pairs,(x,y,z,1),num_image_points]

    def parameter_to_parameter(self,params):
        return params
    
    def parameter_to_point(self,params):
        # 'ZYX'
        _tforms = self.param_to_transform(params)

        return torch.matmul(_tforms, torch.matmul(self.tform_image_pixel_to_mm, self.image_points))[:,:,0:3,:]  # [batch,num_pairs,(x,y,z,1),num_image_points]
    

    
    
    def quaternion_to_transform(self,quaternion): 
        tforms = pytorch3d.transforms.quaternion_to_matrix(quaternion[:,:,0:4])
        transform = torch.cat((tforms,quaternion[...,4:][...,None]),axis=3)
        last_rows = torch.cat((torch.zeros_like(tforms[0,0,:,0]),torch.ones_like(quaternion[0,0,0:1])),axis=0).expand(list(tforms.shape[0:2])[0],list(tforms.shape[0:2])[1],1,4)
        _tforms = torch.cat((
            transform,
            last_rows
            ), axis=2)
        return _tforms
    
    def quaternion_to_parameter(self,quaternion): 
        tforms = self.quaternion_to_transform(quaternion)
        params = self.transform_to_parameter(tforms[:,:,0:3,:].reshape(tforms.shape[0],self.num_pairs,-1))
        return params
    
    def quaternion_to_point(self,quaternion):
        tforms = self.quaternion_to_transform(quaternion)
        points = self.transform_to_point(tforms[:,:,0:3,:].reshape(tforms.shape[0],self.num_pairs,-1))
        return points

    def param_to_transform(self,params):
        # params: (batch,ch,6), "ch": num_pairs, "6":rx,ry,rz,tx,ty,tz
        matrix = pytorch3d.transforms.euler_angles_to_matrix(params[...,0:3], 'ZYX')
        transform = torch.cat((matrix,params[...,3:][...,None]),axis=3)
        last_rows = torch.cat((torch.zeros_like(matrix[0,0,:,0]),torch.ones_like(params[0,0,0:1])),axis=0).expand(list(matrix.shape[0:2])[0],list(matrix.shape[0:2])[1],1,4)
        _tforms = torch.cat((
            transform,
            last_rows
            ), axis=2)
        
        return _tforms



class TransformAccumulation:

    def __init__(
        self, 
        image_points=None,
        tform_image_to_tool=None,
        tform_image_mm_to_tool=None,
        tform_image_pixel_to_image_mm=None
        ):
        self.image_points = image_points
        self.tform_image_to_tool = tform_image_to_tool
        self.tform_image_mm_to_tool = tform_image_mm_to_tool
        self.tform_image_pixel_to_image_mm = tform_image_pixel_to_image_mm
        # pre-compute reference points in tool coordinates
        self.image_points_in_tool = torch.matmul(self.tform_image_to_tool,self.image_points)

        # pre-compute the inverse
        self.tform_tool_to_image = torch.linalg.inv(self.tform_image_to_tool)
        self.tform_tool_to_image_mm = torch.linalg.inv(self.tform_image_mm_to_tool)
        self.tform_image_mm_to_image_pixel = torch.linalg.inv(self.tform_image_pixel_to_image_mm)


    def __call__(self, tform_1_to_0, tform_2_to_1):  
        
        tform_img1_mm_to_img0_mm = tform_1_to_0 # transformation from image1 to image0 in mm
        tform_img2_mm_to_img1_mm = tform_2_to_1
        # transformation from image2 to image0 in mm
        tform_img2_mm_to_img0_mm = torch.matmul(tform_img1_mm_to_img0_mm,tform_img2_mm_to_img1_mm)
        # return point coordinates in image0 in mm, and the transformation from image2 to image0 in mm
        return torch.matmul(tform_img2_mm_to_img0_mm,torch.matmul(self.tform_image_pixel_to_image_mm,self.image_points)),tform_img2_mm_to_img0_mm


class PointTransform:
    def __init__(self,
                label_type=None,
                image_points=None,
                tform_image_to_tool=None,
                tform_image_mm_to_tool = None,
                tform_image_pixel_to_mm = None):
        self.label_type = label_type
        self.tform_image_to_tool = tform_image_to_tool
        self.tform_image_mm_to_tool = tform_image_mm_to_tool
        self.tform_image_pixel_to_mm = tform_image_pixel_to_mm
        self.image_points = image_points
        # pre-compute reference points in tool coordinates
        self.image_points_in_tool = torch.matmul(self.tform_image_to_tool,self.image_points)
        self.tform_tool_to_image = torch.linalg.inv(self.tform_image_to_tool)
        self.tform_tool_to_image_mm = torch.linalg.inv(self.tform_image_mm_to_tool)
        self.tform_image_mm_to_pixel = torch.linalg.inv(self.tform_image_pixel_to_mm)

        if self.label_type=="point":        
            self.call_function = self.point_to_point
        
        elif self.label_type=="transform":  
            self.call_function = self.transform_to_point   

        elif self.label_type=="parameter":
            self.call_function = self.parameter_to_point  

        else:
            raise('Unknown label_type!')
    
    def __call__(self,preds):
        return self.call_function(preds)
        
    def point_to_point(self,pts):
        return pts
    
    def transform_to_point(self,_tforms):
        return torch.matmul(_tforms, torch.matmul(self.tform_image_pixel_to_mm,self.image_points))[:,:,0:3,:]
    
    def parameter_to_point(self,params):
        # 'ZYX'
        _tforms = self.param_to_transform(params)
        # the same as the following function
        # _tforms = pytorch3d.transforms.euler_angles_to_matrix(params, 'ZYX')

        return torch.matmul(_tforms, torch.matmul(self.tform_image_pixel_to_mm,self.image_points))[:,:,0:3,:]  # [batch,num_pairs,(x,y,z,1),num_image_points]
    
    def param_to_transform(self,params):
        # params: (batch,ch,6), "ch": num_pairs, "6":rx,ry,rz,tx,ty,tz
        matrix = pytorch3d.transforms.euler_angles_to_matrix(params[...,0:3], 'ZYX')
        transform = torch.cat((matrix,params[...,3:][...,None]),axis=3)
        last_rows = torch.cat((torch.zeros_like(matrix[0,0,:,0]),torch.ones_like(params[0,0,0:1])),axis=0).expand(list(matrix.shape[0:2])[0],list(matrix.shape[0:2])[1],1,4)
        _tforms = torch.cat((
            transform,
            last_rows
            ), axis=2)
        
        return _tforms
    

class Transform2Transform:
    # transform into 4*4 transformation matrix
    def __init__(self,
                pred_type=None,
                num_pairs=None,
                image_points=None,
                tform_image_to_tool=None,
                tform_image_mm_to_tool = None,
                tform_image_pixel_to_mm = None):
        self.pred_type = pred_type
        self.num_pairs = num_pairs
        self.tform_image_to_tool = tform_image_to_tool
        self.tform_image_mm_to_tool = tform_image_mm_to_tool
        self.tform_image_pixel_to_mm = tform_image_pixel_to_mm
        self.image_points = image_points
        # pre-compute reference points in tool coordinates
        self.image_points_in_tool = torch.matmul(self.tform_image_to_tool,self.image_points)
        self.tform_tool_to_image = torch.linalg.inv(self.tform_image_to_tool)
        self.tform_tool_to_image_mm = torch.linalg.inv(self.tform_image_mm_to_tool)
        self.tform_image_mm_to_pixel = torch.linalg.inv(self.tform_image_pixel_to_mm)


        if self.pred_type=="parameter":
            self.call_function = self.param_to_transform  
        
        elif self.pred_type=="transform":  
            self.call_function = self.transform_to_transform   

        elif self.pred_type=="point":
            
            raise('Not supported!')
        else:
            raise('Unknown label_type!')
        
    def __call__(self, outputs):
        preds = outputs.reshape((outputs.shape[0],self.num_pairs,-1))
        return self.call_function(preds)

    def transform_to_transform(self, transform):
        last_rows = torch.cat([
            torch.zeros_like(transform[..., 0])[..., None, None].expand(-1, -1, 1, 3),
            torch.ones_like(transform[..., 0])[..., None, None]
            ], axis=3)
        transform = torch.cat((
            transform.reshape(-1, self.num_pairs, 3, 4),
            last_rows
            ), axis=2)

        return transform
    
    def param_to_transform(self,params):
        # params: (batch,ch,6), "ch": num_pairs, "6":rx,ry,rz,tx,ty,tz
        matrix = pytorch3d.transforms.euler_angles_to_matrix(params[...,0:3], 'ZYX')
        transform = torch.cat((matrix,params[...,3:][...,None]),axis=3)
        last_rows = torch.cat((torch.zeros_like(matrix[0,0,:,0]),torch.ones_like(params[0,0,0:1])),axis=0).expand(list(matrix.shape[0:2])[0],list(matrix.shape[0:2])[1],1,4)
        _tforms = torch.cat((
            transform,
            last_rows
            ), axis=2)
        
        return _tforms
