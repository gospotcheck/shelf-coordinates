import cv2
from matplotlib import pyplot as plt
import numpy as np

class Color():
    def __init__(self, red, green, blue):
        self.red = red
        self.green = green
        self.blue = blue
    
    def __repr__(self):
        return f"[{self.red}:{self.green}:{self.blue}]"
    
    def __str__(self):
        return f"[{self.red}:{self.green}:{self.blue}]"
    
    def get_color(self):
        return self.red, self.green, self.blue
    
    @staticmethod
    def get_red_color():
        return Color(255,0,0)



def draw_bbox(annotations, image_array, thick = 3):
    '''
    This function draws the bbox, if the match_idx parameter is present 
    it will draw the false positive or false negative boxes according to parameters
    '''
    img_array = np.copy(image_array)

    for box in annotations:
        left = int(box[0])
        top = int(box[1])
        right = int(box[2]+left)
        bottom = int(box[3]+top)
        label = str([box[4],box[5]])
        color = [0, 255, 0]
        cv2.rectangle(img_array,(left, top), (right, bottom), color, thick)
        cv2.putText(img_array, label, (left, top + 60), 0, 1.5, [100, 100, 255], thick)
    return img_array