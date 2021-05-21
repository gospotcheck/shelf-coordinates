import json
import os
import numpy
import cv2
import argparse
import numpy as np
from matplotlib import pyplot as plt
from utils import save_image, toCoco
from relative_coordinates import row_grouping, count_row_objects
from drawing_bbox import draw_bbox

def get_parser():

    my_parser = argparse.ArgumentParser(description='Program to extract bounding box relative coordinates')

    # Add the arguments
    my_parser.add_argument('--img_dir',
                        metavar='Input_Images_Directory',
                        type=str,
                        help='the absolute path to images directory')
    my_parser.add_argument('--ann_path',
                        metavar='input_annotations_file',
                        type=str,
                        help='the absolute path to annotations file')
    my_parser.add_argument('--output_dir',
                        metavar='Output_Result_dir',
                        type=str,
                        help='the absolute path to output results')
    my_parser.add_argument('--threshold_constant',
                        default=0.97,
                        nargs='?',
                        metavar='threshold constant',
                        type=float,
                        help='desired threshold constant to be multiplied with max height')
    return my_parser

def build_coordinate_system(args):
    OUTPUT_DIR_PATH = args.output_dir
    os.makedirs(OUTPUT_DIR_PATH, exist_ok=True)
    img_dir = args.img_dir
    output_dir = os.path.join(OUTPUT_DIR_PATH, "result.json")

    f = open(args.ann_path)
    data = json.load(f)
    f.close()
    annotations_indexes = {}

    for img in os.listdir(img_dir):
        IMG_DIR_PATH = os.path.join(OUTPUT_DIR_PATH, img)
        os.makedirs(IMG_DIR_PATH, exist_ok=True)

        if not img.startswith('.'):
            image_array = cv2.imread(f"{img_dir}/{img}")
            annotations = np.array(data[img]['true'])

            coco_annotations = toCoco(annotations)
            grouped_annotations = row_grouping(coco_annotations, args.threshold_constant)
            count_annotations = count_row_objects(grouped_annotations)
            img_array = draw_bbox(count_annotations, image_array)
            save_image(img_array, os.path.join(IMG_DIR_PATH, "labeled_output.jpg"))

            annotations_indexes[img] = count_annotations

    with open(output_dir, "w") as outfile: 
        json.dump(annotations_indexes, outfile)
    
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    build_coordinate_system(args)