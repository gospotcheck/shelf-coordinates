from nested_dict import nested_dict
from collections import defaultdict
import json
import cv2
import sys
import os
from matplotlib import pyplot as plt
import numpy as np

def save_image(image, path):
    '''
    This function saves the given image to given path
    '''
    cv2.imwrite(path,image)

def calc_iou( gt_bbox, pred_bbox):
    '''
    This function takes the predicted bounding box and ground truth bounding box and 
    return the IoU ratio
    '''
    x_topleft_gt, y_topleft_gt, x_bottomright_gt, y_bottomright_gt= gt_bbox
    x_topleft_p, y_topleft_p, x_bottomright_p, y_bottomright_p= pred_bbox
        
    #if the GT bbox and predcited BBox do not overlap then iou=0
    if(x_bottomright_gt< x_topleft_p):
        return 0.0
    if(y_bottomright_gt< y_topleft_p):  
        return 0.0
    if(x_topleft_gt> x_bottomright_p): 
        return 0.0
    if(y_topleft_gt> y_bottomright_p): 
        return 0.0
    
    GT_bbox_area = (x_bottomright_gt -  x_topleft_gt+1) * (  y_bottomright_gt -y_topleft_gt+1)
    Pred_bbox_area =(x_bottomright_p - x_topleft_p+1) * ( y_bottomright_p -y_topleft_p+1)
    
    x_top_left =np.max([x_topleft_gt, x_topleft_p])
    y_top_left = np.max([y_topleft_gt, y_topleft_p])
    x_bottom_right = np.min([x_bottomright_gt, x_bottomright_p])
    y_bottom_right = np.min([y_bottomright_gt, y_bottomright_p])
    
    intersection_area = (x_bottom_right- x_top_left+1) * (y_bottom_right-y_top_left+1)
    
    union_area = (GT_bbox_area + Pred_bbox_area - intersection_area)
    return intersection_area/union_area

def create_dictionary(tp_idx, fp_idx, fn_idx, confidence_score, img):
    ''' This function creates a global dictionary to store bounding box
    level statistics like which image it belongs to what's the id of that bounding box
    what is the confidence_score and tp fp count'''

    bbox_list = []
    for idx in tp_idx:
        bounding_box_dictionary = {}
        bounding_box_dictionary['img'] = img
        bounding_box_dictionary['detection#'] = idx
        bounding_box_dictionary['Score'] = confidence_score[idx]
        bounding_box_dictionary['tp'] = 1
        bounding_box_dictionary['fp'] = 0
        bbox_list.append(bounding_box_dictionary)
    for idx in fp_idx:
        bounding_box_dictionary = {}
        bounding_box_dictionary['img'] = img
        bounding_box_dictionary['detection#'] = idx
        bounding_box_dictionary['Score'] = confidence_score[idx]
        bounding_box_dictionary['tp'] = 0
        bounding_box_dictionary['fp'] = 1
        bbox_list.append(bounding_box_dictionary)
    return bbox_list

def toCoco(annotations):
    ''' This function is converts the bounding box notations from x_min, y_min, x_max, y_max (Pascal VOC Style) to 
    x_min, y_min, width, height (COCO Style)'''

    for i in range(len(annotations)):
        annotations[i][2] = annotations[i][2]-annotations[i][0]
        annotations[i][3] = annotations[i][3] - annotations[i][1]
    return annotations