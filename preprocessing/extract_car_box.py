"""
    - Script to extract cars box based on annotation
"""
import os
from scipy import io as mat_io
from skimage import io as img_io
import argparse
import traceback


def get_cmd_parser():
    parser = argparse.ArgumentParser(description='Extract Car Box')

    parser.add_argument('-m', '--meta', default='car_dataset/cars_train_annos.mat', type=str,
                      help='cars meta file (default: train)')
    parser.add_argument('-i', '--input', default='car_dataset/cars_train/', type=str,
                      help='input folder')
    parser.add_argument('-o', '--output', default='car_dataset/cars_train_extracted/', type=str,
                      help='output folder')
    return parser

def extract_car(meta_file, input_path, output_path):
    # read meta data and get annotations from meta data
    meta_data = mat_io.loadmat(meta_file)
    anno_data = meta_data['annotations'][0]

    for anno in anno_data:
        # get box position
        bbox_x1 = anno[0][0][0]
        bbox_y1 = anno[1][0][0]

        bbox_x2 = anno[2][0][0]
        bbox_y2 = anno[3][0][0]

        # get image file name
        img_name = anno[-1][0]

        # extract car box from original image
        try:
            in_file = os.path.join(input_path, img_name)
            img_in = img_io.imread(in_file)
        except Exception:
            traceback.print_exc() 
        else:
            out_file = os.path.join(output_path, img_name)
            cars_box = img_in[bbox_y1:bbox_y2, bbox_x1:bbox_x2]
            img_io.imsave(out_file, cars_box)
        

def main():
    # get the parameters of command line 
    parser = get_cmd_parser()
    args = parser.parse_args()

    # get meta file, input folder, output folder
    meta_file = args.meta
    input_path = args.input
    output_path = args.output

    # check if the output folder is created
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    # extract car box
    extract_car(meta_file, input_path, output_path)



if __name__ == '__main__':
    main()
