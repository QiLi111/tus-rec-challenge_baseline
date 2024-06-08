
import torch
import os

def pair_samples(num_samples, num_pred, single_interval):
    """
    :param num_samples:
    :param num_pred: number of the (last) samples, for which the transformations are predicted
        For each "pred" frame, pairs are formed with every one previous frame 
    :param single_interval: 0 - use all interval predictions
                            1,2,3,... - use only specific intervals
    """

    if single_interval == 0:
        return torch.tensor([[n0,n1] for n1 in range(num_samples-num_pred,num_samples) for n0 in range(n1)])
    else:
        return torch.tensor([[n1-single_interval,n1] for n1 in range(single_interval,num_samples,single_interval) ])



def type_dim(label_pred_type, num_points=None, num_pairs=1):
    # return the dimension of the label or prediction, based on the type of label or prediction
    type_dim_dict = {
        "transform": 12,
        "parameter": 6,
        "point": num_points*3,
        "quaternion": 7
    }
    return type_dim_dict[label_pred_type] * num_pairs  # num_points=self.image_points.shape[1]), num_pairs=self.pairs.shape[0]


def save_best_network(opt, model, epoch_label, running_loss_val, running_dist_val, val_loss_min, val_dist_min):
    '''
    :param opt: parameters of this projects
    :param model: model that need to be saved
    :param epoch_label: current epoch
    :param running_loss_val: validation loss of this epoch
    :param running_dist_val: validation distance of this epoch
    :param val_loss_min: min of previous validation losses
    :param val_dist_min: min of previous validation distances
    :return:
    '''

    if running_loss_val < val_loss_min:
        val_loss_min = running_loss_val
        file_name = os.path.join(opt.SAVE_PATH, 'config.txt')
        with open(file_name, 'a') as opt_file:
            opt_file.write('------------ best validation loss result - epoch %s: -------------\n' % (str(epoch_label)))
        if opt.multi_gpu:
            torch.save(model.module.state_dict(), os.path.join(opt.SAVE_PATH, 'saved_model', 'best_validation_loss_model' ))
        else:
            torch.save(model.state_dict(), os.path.join(opt.SAVE_PATH, 'saved_model', 'best_validation_loss_model' ))
        print('Best validation loss parameters saved.')
    
    if running_dist_val < val_dist_min:
        val_dist_min = running_dist_val
        file_name = os.path.join(opt.SAVE_PATH, 'config.txt')
        with open(file_name, 'a') as opt_file:
            opt_file.write('------------ best validation dist result - epoch %s: -------------\n' % (str(epoch_label)))
        
        if opt.multi_gpu:
            torch.save(model.module.state_dict(), os.path.join(opt.SAVE_PATH, 'saved_model', 'best_validation_loss_model' ))
        else:
            torch.save(model.state_dict(), os.path.join(opt.SAVE_PATH, 'saved_model', 'best_validation_dist_model'))
        print('Best validation dist parameters saved.')
    
    return val_loss_min, val_dist_min



def add_scalars(writer,epoch, loss_dists):
    # loss and average distance in training and val
    train_epoch_loss = loss_dists['train_epoch_loss']
    train_epoch_dist = loss_dists['train_epoch_dist'].mean()
    epoch_loss_val = loss_dists['val_epoch_loss']
    epoch_dist_val = loss_dists['val_epoch_dist'].mean()
    
    writer.add_scalars('loss', {'train_loss': train_epoch_loss},epoch)
    writer.add_scalars('loss', {'val_loss': epoch_loss_val},epoch)
    writer.add_scalars('dist', {'train_dist': train_epoch_dist}, epoch)
    writer.add_scalars('dist', {'val_dist': epoch_dist_val}, epoch)

def write_to_txt(opt,epoch, loss_dists):
    # write loss and average distance in training and val to txt
    train_epoch_loss = loss_dists['train_epoch_loss']
    train_epoch_dist = loss_dists['train_epoch_dist'].mean()
    epoch_loss_val = loss_dists['val_epoch_loss']
    epoch_dist_val = loss_dists['val_epoch_dist'].mean()
    
    file_name_train = os.path.join(opt.SAVE_PATH, 'train_results', 'train_loss.txt')
    with open(file_name_train, 'a') as opt_file_train:
        print('[Epoch %d], train-loss=%.3f, train-dist=%.3f' % (epoch, train_epoch_loss, train_epoch_dist),file=opt_file_train)

    file_name_val = os.path.join(opt.SAVE_PATH, 'val_results', 'val_loss.txt')
    with open(file_name_val, 'a') as opt_file_val:
        print('[Epoch %d], val-loss=%.3f, val-dist=%.3f' % (epoch, epoch_loss_val, epoch_dist_val), file=opt_file_val)

def print_info(epoch,loss,dist,opt,train_val):
    # print loss and average distance in training and val
    if epoch in range(int(opt.retrain_epoch), int(opt.retrain_epoch)+opt.NUM_EPOCHS, opt.FREQ_INFO):
        if train_val == 'train':
            print('[Epoch %d] train-loss=%.3f, train-dist=%.3f' % (epoch, loss, dist.mean()))
        elif train_val == 'val':
            print('[Epoch %d] val-loss=%.3f, val-dist=%.3f' % (epoch, loss, dist.mean()))

        if dist.shape[0]>1: # torch.tensor([dist]).shape[0]>1
            print('%.2f '*dist.shape[0] % tuple(dist))

def save_model(model,epoch,opt):
    # save the model at current epoch, and keep the number of models at 4
    if epoch in range(int(opt.retrain_epoch), int(opt.retrain_epoch)+opt.NUM_EPOCHS, opt.FREQ_SAVE):
               
        if opt.multi_gpu:
            torch.save(model.module.state_dict(), os.path.join(opt.SAVE_PATH, 'saved_model', 'model_epoch%08d' % epoch))
        else:
            torch.save(model.state_dict(), os.path.join(opt.SAVE_PATH, 'saved_model', 'model_epoch%08d' % epoch))
       
        print('Model parameters saved.')
        list_dir = os.listdir(os.path.join(opt.SAVE_PATH, 'saved_model'))
        saved_models = [i for i in list_dir if i.startswith('model_epoch')]
        if len(saved_models)>4:
            os.remove(os.path.join(opt.SAVE_PATH,'saved_model',sorted(saved_models)[0]))
