import skimage.io as io
import numpy as np
import h5py
from torch.utils.data import Dataset, DataLoader  # For custom data-sets
import torch 
from PIL import Image
import torchvision.transforms as transforms
import sys 
sys.path.append("..")
from graphs import plot_segmentation


class NYU_V2_Dataset(Dataset):
    def __init__(
        self,
        data_path,
        n_class=41,
        in_size=(640, 480),
        out_size=(640, 480),
        max_depth=0,
    ):
        f = h5py.File(data_path, 'r')
        self.data = f
        self.n_class = n_class
        self.in_size = in_size
        self.out_size = out_size
        self.data_path = data_path

        self.resize_img = transforms.Compose(
            [
                transforms.Resize(size=out_size),
                transforms.ToTensor(),
                # mean and std values calculated over training data
                transforms.Normalize(
                    (0.48253979, 0.41267643, 0.39449769), (0.36771, 0.36445, 0.36256)
                ),
            ]
        )

        self.resize_label = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(size=out_size, interpolation=Image.NEAREST),
                transforms.ToTensor(),
            ]
        )

        self.max_depth = max_depth

    def __len__(self):
        return len(self.data['images'])

    def __getitem__(self, idx):
        img = self.data['images'][idx]
        depth = self.data['depths'][idx]
        semseg = np.array(self.data['labels'][idx]).astype("float32")
        
        img = np.transpose(img, (2,1,0))
        img = Image.fromarray(img, 'RGB')
        semseg = np.transpose(semseg, (1, 0))
        semseg[semseg > 40] = 0
        depth = np.transpose(depth, (1, 0))
        if self.max_depth: depth[depth != depth] = self.max_depth
        
        #depth = np.array( h5py.File(self.data_path + depth_path, "r").get("dataset") ).astype("float32")
        #semseg = np.array(h5py.File(self.data_path + semseg_path, "r").get("dataset"))
        #semseg[semseg == -1] = 0  # -1 is used in segmentation to denote unlabelled (0 is unused)

        # instseg = np.array(h5py.File(self.data_path + instseg_path, 'r').get('dataset')) #int16
        # print(depth.dtype)

        img = self.resize_img(img)
        depth, semseg = ( self.resize_label(depth).squeeze(0), self.resize_label(semseg).to(torch.long).squeeze(0),)

        #img, semseg, depth = self.augment(img, semseg, depth)
        
        ## In some older versions of torch, applying transforms.ToTensor() would rescale the values
        ## to be between 0, 1. In this case, the output tensor should be multiplied by 255. In
        ## the case of semantic labels, the tensor should also be converted to type 'long'/'int64'

        return img, depth, semseg

    def augment(self, image, mask, depth):
        rand_augment = np.random.randint(5)

        if rand_augment == 0:
            crop_size = [int(s*0.75) for s in self.in_size]
            x, y, h, w = transforms.RandomCrop.get_params(image, output_size=crop_size)
            image = transforms.functional.crop(image, x, y, h, w)
            mask = transforms.functional.crop(mask, x, y, h, w)
            depth = transforms.functional.crop(depth, x, y, h, w)
        elif rand_augment == 1:
            image = transforms.functional.vflip(image)
            mask = transforms.functional.vflip(mask)
            depth = transforms.functional.vflip(depth)
        elif rand_augment == 2:
            rot_range = (-20,20)
            rot_angle = transforms.RandomRotation.get_params(rot_range)
            image = transforms.functional.rotate(image, rot_angle, resample = Image.BILINEAR)
            mask = transforms.functional.rotate(mask, rot_angle, resample = Image.NEAREST)
            depth = transforms.functional.rotate(depth, rot_angle, resample = Image.NEAREST)
        
        #2/5 chance no augmentation
        
        return image, mask, depth

if __name__ == "__main__":
    data = NYU_V2_Dataset('./NYU_data/test_data_baseline.h5')
    dataloader = DataLoader(data, batch_size=32,num_workers=0, shuffle=False)
    img, depth, seg = next(iter(dataloader))
    img, depth, seg = img[11], depth[11], seg[11]

    plot_segmentation(img, seg, "./seg.png", title='Seg')
    plot_segmentation(img, depth, "./depth.png", title='depth')


