import glob, os
from torchvision import transforms
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


class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, mask=None):
        image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        if mask is None:
            return image
        mask = cv2.resize(mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return image, mask


class ImageToTensor(object):
    def __init__(self, normalize=True):
        self.normalize = normalize

    def __call__(self, img):
        img = torch.from_numpy(img.transpose(2, 0, 1).copy())
        if self.normalize:
            return img.float().div(255)
        else:
            return img.float()


class TestDataset(Dataset):
    def __init__(self, test_image_dir, test_image_datasets, test_size):
        self.test_image_dir = test_image_dir
        self.test_image_datasets = test_image_datasets
        self.test_size = test_size

        self.dataset_name = []
        self.images = []

        for dataset in self.test_image_datasets:
            self.dataset_name += [dataset]
            self.images += glob.glob(os.path.join(self.test_image_dir, dataset, r"img/*.jpg"))
        print(f"current test dataset {'+'.join(self.dataset_name)} : {len(self.images)} images")

        self.img_transform = transforms.Compose([
            Resize(self.test_size[0], self.test_size[1]),
            ImageToTensor(normalize=True),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = cv2.imread(img_path)
        img = img[:, :, ::-1]

        # save_path: /DATASET/xxx.png
        save_path = img_path.split(os.path.sep)
        save_path = os.path.sep.join(save_path[-3:]).replace("img" + os.path.sep, "").replace(".jpg", ".png")

        size = [img.shape[0], img.shape[1]]

        img = self.img_transform(img)
        return img, torch.tensor(size), save_path

    def __len__(self):
        return len(self.images)
