#%%
from torch.utils.data import Dataset, DataLoader  # For custom data-sets
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import os
import numpy as np
from PIL import Image
import torch
import pandas as pd
from collections import namedtuple
import h5py
import matplotlib.pyplot as plt
import time
import random


n_class = 41

# TODO change path to your hypersim_data directory
DATA_PATH = "./data/"

# a label and all meta information
Label = namedtuple(
    "Label",
    [
        "name",
        "semantic_id",
        "color",
    ],
)

labels = [
    #       name            id       color
    Label("unknown", 0, (0, 0, 0)),
    Label("wall", 1, (174, 199, 232)),
    Label("floor", 2, (152, 223, 138)),
    Label("cabinet", 3, (31, 119, 180)),
    Label("bed", 4, (255, 187, 120)),
    Label("chair", 5, (188, 189, 34)),
    Label("sofa", 6, (140, 86, 75)),
    Label("table", 7, (255, 152, 150)),
    Label("door", 8, (214, 39, 40)),
    Label("window", 9, (197, 176, 213)),
    Label("bookshelf", 10, (148, 103, 189)),
    Label("picture", 11, (196, 156, 148)),
    Label("counter", 12, (23, 190, 207)),
    Label("blinds", 13, (178, 76, 76)),
    Label("desk", 14, (247, 182, 210)),
    Label("shelves", 15, (66, 188, 102)),
    Label("curtain", 16, (219, 219, 141)),
    Label("dresser", 17, (140, 57, 197)),
    Label("pillow", 18, (202, 185, 52)),
    Label("mirror", 19, (51, 176, 203)),
    Label("floormat", 20, (200, 54, 131)),
    Label("clothes", 21, (92, 193, 61)),
    Label("ceiling", 22, (78, 71, 183)),
    Label("books", 23, (172, 114, 82)),
    Label("refrigerator", 24, (255, 127, 14)),
    Label("television", 25, (91, 163, 138)),
    Label("paper", 26, (153, 98, 156)),
    Label("towel", 27, (140, 153, 101)),
    Label("showercurtain", 28, (158, 218, 229)),
    Label("box", 29, (100, 125, 154)),
    Label("whiteboard", 30, (178, 127, 135)),
    Label("person", 31, (120, 185, 128)),
    Label("nightstand", 32, (146, 111, 194)),
    Label("toilet", 33, (44, 160, 44)),
    Label("sink", 34, (112, 128, 144)),
    Label("lamp", 35, (96, 207, 209)),
    Label("bathtub", 36, (227, 119, 194)),
    Label("bag", 37, (213, 92, 176)),
    Label("otherstruct", 38, (94, 106, 211)),
    Label("otherfurniture", 39, (82, 84, 163)),
    Label("otherprop", 40, (100, 85, 144)),
]


class ToNumpy(object):
    def __call__(self, sample):
        return np.array(sample)


class HypersimDataset(Dataset):
    def __init__(
        self,
        csv_file,
        n_class=n_class,
        in_size=(768, 1024),
        out_size=(192, 256),
        data_path=DATA_PATH,
        max_depth=0,
        use_transform=False,
    ):
        self.data = pd.read_csv(csv_file, index_col=0)
        self.data.reset_index(inplace=True)
        self.n_class = n_class
        self.mode = csv_file
        self.in_size = in_size
        self.out_size = out_size
        self.data_path = data_path
        if not use_transform:
            self.resize_img = transforms.Compose(
                [
                    transforms.Resize(size=out_size, interpolation=Image.BICUBIC),
                    transforms.ToTensor(),
                    # mean and std values calculated over training data
                    transforms.Normalize(
                        (0.42228, 0.38978, 0.35852), (0.36771, 0.36445, 0.36256)
                    ),
                ]
            )

            self.resize_label = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize(size=out_size, interpolation=Image.NEAREST),
                    ToNumpy(),
                    transforms.ToTensor(),
                ]
            )
            self.resize_depth = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize(size=out_size, interpolation=Image.BILINEAR),
                    ToNumpy(),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.resize_img = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    # mean and std values calculated over training data
                    transforms.Normalize(
                        (0.42228, 0.38978, 0.35852), (0.36771, 0.36445, 0.36256)
                    ),
                ]
            )

            self.resize_label = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.RandomHorizontalFlip(),
                    # transforms.CenterCrop(448),
                    ToNumpy(),
                    transforms.ToTensor(),
                ]
            )

            self.resize_depth = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.RandomHorizontalFlip(),
                    # transforms.CenterCrop(448),
                    ToNumpy(),
                    transforms.ToTensor(),
                ]
            )

        self.max_depth = max_depth

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.loc[idx, "Image Path"]
        depth_path = self.data.loc[idx, "Depth Path"]
        semseg_path = self.data.loc[idx, "Semantic Segmentation Path"]
        instseg_path = self.data.loc[idx, "Instance Segmentation Path"]

        img = Image.open(self.data_path + img_path).convert("RGB")

        depth = np.array(
            h5py.File(self.data_path + depth_path, "r").get("dataset")
        ).astype("float32")
        semseg = np.array(h5py.File(self.data_path + semseg_path, "r").get("dataset"))
        semseg[
            semseg == -1
        ] = 0  # -1 is used in segmentation to denote unlabelled (0 is unused)

        # instseg = np.array(h5py.File(self.data_path + instseg_path, 'r').get('dataset')) #int16
        # print(depth.dtype)
        seed = int(time.time())
        random.seed(seed)
        torch.manual_seed(seed)  # needed for torchvision 0.7
        img = self.resize_img(img)
        random.seed(seed)
        torch.manual_seed(seed)  # needed for torchvision 0.7
        depth = self.resize_depth(depth).squeeze(0)
        random.seed(seed)
        torch.manual_seed(seed)  # needed for torchvision 0.7
        semseg = self.resize_label(semseg).to(torch.long).squeeze(0)
        if self.max_depth:
            depth[depth != depth] = self.max_depth
        ## In some older versions of torch, applying transforms.ToTensor() would rescale the values
        ## to be between 0, 1. In this case, the output tensor should be multiplied by 255. In
        ## the case of semantic labels, the tensor should also be converted to type 'long'/'int64'

        return img, depth, semseg


#%%

# class mapping
# https://github.com/ankurhanda/SceneNetv1.0/
# class 0 is unknown
class NYUDataset(Dataset):
    def __init__(
        self,
        dataset_type="train",
        n_class=n_class,
        in_size=(480, 640),
        out_size=(240, 320),
        data_path=DATA_PATH,
        max_depth=0,
        use_transform=False,
    ):
        assert dataset_type in ["train", "test"]
        if dataset_type == "test":
            dataset_type = "val"
        self.dataset_type = dataset_type
        self.image_names = np.loadtxt(
            os.path.join(data_path, "NYUD_MT", "gt_sets", f"{dataset_type}.txt"),
            dtype=str,
        )
        self.n_class = n_class
        self.out_size = out_size
        self.data_path = data_path
        self.use_transform = use_transform
        self.normalize = transforms.Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        )

        if not use_transform:
            self.resize_img = transforms.Compose(
                [
                    transforms.Resize(size=out_size, interpolation=Image.BILINEAR),
                    transforms.ToTensor(),
                    # mean and std values calculated over training data
                ]
            )

            self.resize_label = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize(size=out_size, interpolation=Image.NEAREST),
                    ToNumpy(),
                    transforms.ToTensor(),
                ]
            )
            self.resize_depth = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize(size=out_size, interpolation=Image.NEAREST),
                    ToNumpy(),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.resize_img = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            )

            self.resize_label = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.RandomHorizontalFlip(),
                    ToNumpy(),
                    transforms.ToTensor(),
                ]
            )

            self.resize_depth = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.RandomHorizontalFlip(),
                    ToNumpy(),
                    transforms.ToTensor(),
                ]
            )

        self.max_depth = max_depth

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_path = os.path.join(
            self.data_path,
            "NYUD_MT",
            "images",
            self.image_names[idx],
        )
        depth_path = os.path.join(
            self.data_path, "NYUD_MT", "depth", self.image_names[idx]
        )
        semseg_path = os.path.join(
            self.data_path, "NYUD_MT", "segmentation", self.image_names[idx]
        )

        img = Image.open(img_path + ".jpg")
        depth = np.load(depth_path + ".npy")
        semseg = np.array(Image.open(semseg_path + ".png"), dtype=np.int32)

        seed = int(time.time())
        random.seed(seed)
        torch.manual_seed(seed)  # needed for torchvision 0.7
        img = self.resize_img(img)

        random.seed(seed)
        torch.manual_seed(seed)  # needed for torchvision 0.7
        depth = self.resize_depth(depth).squeeze(0)

        random.seed(seed)
        torch.manual_seed(seed)  # needed for torchvision 0.7
        semseg = self.resize_label(semseg).to(torch.long).squeeze(0)

        if self.use_transform and random.random() > 0.5:
            # if self.use_transform:
            gamma = random.uniform(0.9, 1.1)
            img = img ** gamma
            brightness = random.uniform(0.75, 1.25)
            img = img * brightness
            colors = np.random.uniform(0.9, 1.1, size=3)
            img = img * colors[:, None, None]
            img = img.clamp(0, 1)
            img = img.to(torch.float32)
        img = self.normalize(img)

        if self.max_depth:
            depth[depth != depth] = self.max_depth
        return img, depth, semseg


# %%
if __name__ == "__main__":
    dataset = NYUDataset("test", use_transform=True)
    img, depth, semseg = dataset[0]
    # plt.imshow(semseg)
    # plt.imshow(depth)
    plt.imshow(img.permute(1, 2, 0))
    # for i, (img, depth, semseg) in enumerate(dataset):
    #     if (depth == 0).sum() > 0:
    #         print(i)
    #         break
    # img, depth, semseg = dataset[2]

#%%
