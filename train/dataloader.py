import numpy as np
import glob,os
import random
import cv2
import torch
from torch.utils.data import Dataset
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *img):
        for t in self.transforms:
            img = t(*img)
        return img


class RandomFlip(object):
    def __call__(self, image, mask):
        if np.random.randint(2) == 0:
            return image[:, ::-1, :], mask[:, ::-1]
        else:
            return image, mask


def collate(batch):
    size = [224, 256, 288, 320, 352][random.randint(0, 4)]
    image, mask = [list(item) for item in zip(*batch)]
    for i in range(len(batch)):
        image[i] = (cv2.resize(image[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)-np.array([[[124.55, 118.90, 102.94]]]))/np.array([[[56.77, 55.97, 57.50]]])
        mask[i] = cv2.resize(mask[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
    image = torch.from_numpy(np.stack(image, axis=0)).permute(0, 3, 1, 2).contiguous().float() #[B,3,H,W]
    mask = torch.from_numpy(np.stack(mask, axis=0)).unsqueeze(1).float()/255 #[B,1,H,W]
    return image, mask


class TrainDataset(Dataset):
    def __init__(self, train_image_dir, train_image_datasets):
        #images : /path/to/train/dataset/img/*.jpg
        #labels : /path/to/train/dataset/mask/*.png
        self.train_image_dir = train_image_dir
        self.train_image_datasets = train_image_datasets

        #record all train samples
        self.images = []

        self.dataset_name = []
        for dataset in self.train_image_datasets:
            self.dataset_name += [dataset]
            self.images += glob.glob(os.path.join(self.train_image_dir, dataset,r"img/*.jpg"))
        print(f"current image dataset {'+'.join(self.dataset_name)} : {len(self.images)} images")
        
        self.joint_transform = Compose([RandomFlip()])

    def __getitem__(self, idx):
        path = self.images[idx]
        label = path.replace("img", "mask").replace(".jpg", ".png")

        image = cv2.imread(path)
        image = image[:,:,::-1] #BGR2RGB
        label = cv2.imread(label, cv2.IMREAD_GRAYSCALE)

        #random flip at training
        image, label = self.joint_transform(image, label)
        return image, label

    def __len__(self):
        return len(self.images)
