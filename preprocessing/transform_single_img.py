"""
    - Script to transform a single image
"""
import os
import torchvision
from torchvision import transforms
from skimage import io
from PIL import Image
import argparse


def get_cmd_parser():
    parser = argparse.ArgumentParser(description='Single Image transform')
    parser.add_argument('-i', '--input', default='small_car_dataset/cars_train_extracted/00006.jpg', type=str,
                      help='the image file to be transformed')
    parser.add_argument('-w', '--width', default=224, type=int,
                      help='the width of resized image')
    parser.add_argument('-t', '--height', default=224, type=int,
                      help='the height of resized image')
    return parser

# Define the transformer for image data
def get_transformer(width, height, mean, std):
    tfms = transforms.Compose([
        # 1. Resize to make sure all images having the same size 
        transforms.Resize((width, height)),  

        # 2. Horizontal flip   
        transforms.RandomHorizontalFlip(),

        # 3. Rotation
        transforms.RandomRotation(15),

        # 4. Save into tensor
        transforms.ToTensor(),

        # 5. Normalization
        transforms.Normalize(mean=mean, std=std)
    ])

    return tfms

def transform_img(img_file, tfms):
    # Open image file
    image = Image.open(img_file)

    # Transform the image
    image_tf = tfms(image)

    # Save the image
    tgt_img = '/content/drive/MyDrive/data/small_car_dataset/00001_tfms.jpg'
    file_name = os.path.basename(img_file)
    file_path = os.path.dirname(img_file)
    tgt_img_file = os.path.join(file_path, 'tfms_'+file_name)
    to_pil_image = transforms.ToPILImage()
    pil_image = to_pil_image(image_tf)
    pil_image.save(tgt_img_file)


def main():
    # get the parameters of command line 
    parser = get_cmd_parser()
    args = parser.parse_args()

    # Set the image size or use the inputed parameters
    width, height = args.width, args.height

    # Set the parameters for normalization
    mean = [0.485, 0.456, 0.406]  # Mean values for normalization (from ImageNet)
    std = [0.229, 0.224, 0.225]   # Standard deviation values for normalization (from ImageNet)

    # Get the transformer
    tfms = get_transformer(width, height, mean, std)

    # Transform the image and save
    img_file = args.input
    transform_img(img_file, tfms)


if __name__ == '__main__':
    main()
