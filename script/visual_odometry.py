import cv2
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import yaml
from dataloader import DataLoader
from utils import stereo_depth


# Driver Code
if __name__ == '__main__':

    # Load Config File
    with open("../config/initial_config.yaml", "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as error:
            print(error)

    # Declare Necessary Variables
    sequence = config['data']['sequence']

    # Create Instances
    data_handler = DataLoader(sequence=sequence)

    # Local variables
    left_image = data_handler.first_image_left
    right_image = data_handler.first_image_right
    left_camera_matrix = data_handler.P0
    right_camera_matrix = data_handler.P1

    # Visualise Ground Truth
    # print(data_handler.gt_visualisation())

    # # To know matrices of each frame wrt. to the world coordinate
    # print(
    #     f'The translation of that frame wrt. world coordinate \n{data_handler.gt_translation_matrix()}\n')
    # print(
    #     f'The rotation of that frame wrt. world coordinate \n{data_handler.gt_rotation_matrix()}\n')

    depth_map, disp_map = stereo_depth(left_image, right_image,
                                       left_camera_matrix, right_camera_matrix)

    plt.figure(figsize=(14, 8))
    plt.title('Stereo Depth Mapping')
    plt.imshow(depth_map)

    plt.figure(figsize=(14, 8))
    plt.title('Disparity Map of the Left Camera')
    plt.imshow(disp_map)
    plt.show()
