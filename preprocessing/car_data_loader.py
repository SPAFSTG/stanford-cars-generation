import os
import numpy as np
from scipy import io as mat_io
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import argparse


class CarsDataset(Dataset):
    """
        Cars Dataset
    """
    def __init__(self, data_dir, meta_file, resize_width, resize_height, limit=None):
        self.data = []
        self.resize_width = resize_width
        self.resize_height = resize_height

        if not isinstance(meta_file, str):
            raise Exception("meta file must be string location !")
        
        # Read meta data from .mat file
        meta_data = mat_io.loadmat(meta_file)
        annotations = meta_data['annotations'][0]

        # Add img file to data list
        for idx, anno in enumerate(annotations):
            if limit is not None and idx >= limit:
                break

            img_filename = anno[-1][0]
            img_filepath = os.path.join(data_dir, img_filename)
            self.data.append(img_filepath)

        # Setup the transform
        mean = [0.485, 0.456, 0.406]  # Mean values for normalization (from ImageNet)
        std = [0.229, 0.224, 0.225]   # Standard deviation values for normalization (from ImageNet)
        self.transform = transforms.Compose([
            # 1. Resize to make sure all images having the same size 
            transforms.Resize((self.resize_width, self.resize_height)),  
            # 2. Horizontal flip   
            transforms.RandomHorizontalFlip(),
            # 3. Rotation
            transforms.RandomRotation(15),
            # 4. Save into tensor
            transforms.ToTensor(),
            # 5. Normalization
            transforms.Normalize(mean=mean, std=std)
        ])

    def __getitem__(self, idx):
        # Open image file
        img_file = self.data[idx]
        img = Image.open(img_file)

        # Transform the image
        img_tfms = self.transform(img)

        return img_tfms

    def __len__(self):
        return len(self.data)


class CarsDataLoader(DataLoader):
    """
    Cars data loading
    """
    def __init__(self, data_dir, meta_file, resize_width=224,resize_height=224, limit=None,
                 batch_size=32, num_workers=1, shuffle=True):

        self.dataset = CarsDataset(data_dir, meta_file, resize_width, resize_height, limit)
        super(CarsDataLoader, self).__init__(self.dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)


if __name__ == '__main__':
    # Create cmd parameter parser
    parser = argparse.ArgumentParser(description='Extract Car Box')

    parser.add_argument('-d', '--datadir', default='car_dataset/cars_train_extracted/', type=str,
                      help='data dir storing the image files')
    parser.add_argument('-m', '--meta', default='car_dataset/cars_train_annos.mat', type=str,
                      help='cars meta file (default: train)')
    parser.add_argument('-w', '--width', default=224, type=int,
                      help='the width of resized image')
    parser.add_argument('-t', '--height', default=224, type=int,
                      help='the height of resized image')
    parser.add_argument('-l', '--limit', default=10, type=int,
                      help='limit the maximum number of image files being loaded')

    # Get command line parameters
    args = parser.parse_args()
    data_dir = args.datadir
    meta_file = args.meta
    resize_width = args.width
    resize_height =  args.height
    limit = args.limit

    # Create the data loader for car image
    car_data_loder = CarsDataLoader(data_dir, meta_file, resize_width, resize_height, limit=limit)

    # Get img data from data_loader
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for batch_idx, (data) in enumerate(car_data_loder):
        data = data.to(device)
        print('success to read batch data, batch index:', batch_idx)
        print('data length for this batch is:', len(data))

