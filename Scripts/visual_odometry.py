import cv2
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import yaml
from utils import CameraPoses


# Driver Code
if __name__ == '__main__':

    with open("../config/initial_config.yaml", "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as error:
            print(error)

    pose_filepath = config['data']['pose_path']
    calib_filepath = config['data']['calib_path']

    #################### Ground Truth #########################
    ground_truth = CameraPoses.data_description(pose_filepath)
    print(f'{ground_truth}\n')

    ################### Visualisation ########################
    ground_truth = CameraPoses.visualisation(pose_filepath)

    # To know metrices of each frame wrt. to the world coordinate
    any_gt = CameraPoses(position=0)
    print(
        f'The translation of that frame wrt. world coordinate \n{any_gt.translation_matrix()}\n')
    print(
        f'The rotation of that frame wrt. world coordinate \n{any_gt.rotation_matrix()}\n')

    #################### Ground Truth #########################
