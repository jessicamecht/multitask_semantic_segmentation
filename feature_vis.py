"""
Created on Thu Oct 26 11:23:47 2017
@author: Utku Ozbulak - github.com/utkuozbulak
"""
import torch
from torch.nn import ReLU

from unet.multitask_unet import UNet_Multitask
from dataloader import * 
from experiments.experiment_unet_multitask import *
from misc_functions import (convert_to_grayscale,
                            save_gradient_images,
                            get_positive_negative_saliency)


class GuidedBackprop():
    """
       Produces gradients generated with guided back propagation from the given image
    """
    def __init__(self, model, layer):
        self.model = model
        self.gradients = None
        self.layer = layer
        self.forward_relu_outputs = []
        # Put model in evaluation mode
        self.model.eval()
        self.update_relus()
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]
        # Register hook to the first layer
        self.layer.register_backward_hook(hook_function)

    def update_relus(self):
        """
            Updates relu activation functions so that
                1- stores output in forward pass
                2- imputes zero for gradient values that are less than zero
        """
        def relu_backward_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, change it to zero
            """
            # Get last forward output
            corresponding_forward_output = self.forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)
            del self.forward_relu_outputs[-1]  # Remove last forward output
            return (modified_grad_out,)

        def relu_forward_hook_function(module, ten_in, ten_out):
            """
            Store results of forward pass
            """
            self.forward_relu_outputs.append(ten_out)

        # Loop through layers, hook up ReLUs
        for pos, module in self.model.encoder._modules.items():
            if isinstance(module, ReLU):
                module.register_backward_hook(relu_backward_hook_function)
                module.register_forward_hook(relu_forward_hook_function)

    def generate_gradients(self, input_image, target_class, depth_bp=True, class_nb=None):
        # Forward pass
        model_output, model_output_depth = self.model(input_image)
        # Zero gradients
        self.model.zero_grad()
        # Backward pass
        if depth_bp:
            model_output_depth.backward(gradient=target_class)
        else:
            pred = model_output[:,class_nb,:,:]
            pred.backward(gradient=target_class)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = self.gradients.data.numpy()[0]
        return gradients_as_arr


if __name__ == '__main__':
    print('Guided backprop started')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    exp_name = "experiment_unet_multitask"
    config_dict = read_file(exp_name)
    hparams = dict_to_args(config_dict)
    pretrained_model = UNet_Multitask_lightning(hparams, 3, 40 + 1)
    PATH = './experiment_unet_multitask/v_03/checkpoints/epoch=0-step=571.ckpt' #change checkpoint if necessary
    checkpoint = torch.load(PATH, map_location=torch.device(device))
    
    pretrained_model.load_state_dict(checkpoint['state_dict'])
    csv = "data/train_data.csv"
    train_data = HypersimDataset(csv_file=csv, data_path = "./data/")
    train_loader = DataLoader(train_data, batch_size=1,num_workers=0, shuffle=True)
    #first layer to hook gradient vis onto
    layer = list(pretrained_model.model.encoder.inc.double_convolution._modules.items())[0][1]

    prep_img, depth, semseg = next(iter(train_loader))
    file_name_to_export = './'
    prep_img.requires_grad = True

    # Guided backprop
    GBP = GuidedBackprop(pretrained_model.model, layer)
    # Get gradients for depth 
    guided_grads = GBP.generate_gradients(prep_img, depth.squeeze())
    # Save colored gradients
    save_gradient_images(guided_grads, file_name_to_export + '_Guided_BP_color')
    # Convert to grayscale
    grayscale_guided_grads = convert_to_grayscale(guided_grads)
    # Save grayscale gradients
    save_gradient_images(grayscale_guided_grads, file_name_to_export + '_Guided_BP_gray')
    # Positive and negative saliency maps
    pos_sal, neg_sal = get_positive_negative_saliency(guided_grads)
    save_gradient_images(pos_sal, file_name_to_export + '_pos_sal')
    save_gradient_images(neg_sal, file_name_to_export + '_neg_sal')


    ##### get gradient for segmentation for most occurring class
    unique, counts = torch.unique(semseg, sorted=True, return_counts=True)
    max_idx = torch.argmax(counts)
    class_nb = unique[max_idx]
    guided_grads = GBP.generate_gradients(prep_img, semseg, depth_bp=False, class_nb=class_nb)
    # Save colored gradients
    save_gradient_images(guided_grads, file_name_to_export + '_Guided_BP_color_seg')
    # Convert to grayscale
    grayscale_guided_grads = convert_to_grayscale(guided_grads)
    # Save grayscale gradients
    save_gradient_images(grayscale_guided_grads, file_name_to_export + '_Guided_BP_gray_seg')
    # Positive and negative saliency maps
    pos_sal, neg_sal = get_positive_negative_saliency(guided_grads)
    save_gradient_images(pos_sal, file_name_to_export + '_pos_sal_seg')
    save_gradient_images(neg_sal, file_name_to_export + '_neg_sal_seg')

    print('Guided backprop completed')