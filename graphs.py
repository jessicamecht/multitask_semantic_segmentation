import numpy as np
import matplotlib.pyplot as plt
import os

class UnNormalize(object):
    '''to make our image look nice we need to unnormalize it'''
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        '''Args: tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns: Tensor: Normalized image.'''
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

def plot_segmentation(image, prediction, file_path, title='title'):
    '''plots segmentation mask on top pf the image
    param: image tensor with size (channels, width, height)
    param: prediction tensor with size (width, height)
    param: title String
    param: filename String'''

    unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    image = unorm(image).cpu()
    prediction = prediction.cpu()
    image = (np.array(image.permute(1, 2, 0)) * 255).astype(int)
    plt.imshow(image)
    plt.imshow(prediction, cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.title(title)
    plt.savefig(file_path)
    plt.close()
    
    
def save_results(image, seg_pred, depth_pred, true_depth, true_seg, root_dir, img_name):             
    '''plots segmentation mask on top pf the image
    param: image tensor with size (channels, width, height)
    param: prediction tensor with size (width, height)
    param: title String
    param: filename String'''

    unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    image = unorm(image).cpu()
    seg_pred = seg_pred.cpu().detach().numpy()
    depth_pred = depth_pred.cpu().detach().numpy()
    image = (np.array(image.permute(1, 2, 0)) * 255).astype(int)
    
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
    save_dir = os.path.join(root_dir, img_name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        
    img_path = f'{save_dir}/image.png'
    seg_path = f'{save_dir}/seg_mask.png'
    depth_path = f'{save_dir}/depth_map.png'
    true_seg_path = f'{save_dir}/true_seg.png'
    true_depth_path = f'{save_dir}/true_depth.png'
    save_image(image, img_path)
    save_image(depth_pred, depth_path)
    save_segmentation_mask(image, true_seg, true_seg_path)
    save_image(true_depth, true_depth_path)
    save_segmentation_mask(image, seg_pred, seg_path)
    
def save_image(image, file_path):
    plt.imshow(image)
    plt.axis('off')
    plt.savefig(file_path)
    plt.close()

def save_segmentation_mask(image, seg_pred, file_path):
    plt.imshow(image)
    plt.imshow(seg_pred, cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.savefig(file_path)
    plt.close()

    