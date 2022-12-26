import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

with open("../config/initial_config.yaml", "r") as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


pose_path = config['data']['pose_path']


# To know the pose of the camera wrt. the world coordinate
class CameraPoses:

    # Read the pose data from the text file
    cam_pose = pd.read_csv(pose_path, delimiter=' ', header=None)

    """
        params: position = frame number
    """

    def __init__(self, position):
        self.position = position

    # Augmented Rotation and Translated Matrix
    def postion_matrix(self):
        location = np.array(
            self.cam_pose.iloc[self.position]).reshape(3, 4).round(3)
        return location

    # Rotation Matrix
    def rotation_matrix(self):
        rotation = self.postion_matrix()[0:, 0:3]
        return rotation

    # Translation Matrix
    def translation_matrix(self):
        translation = self.postion_matrix()[0:, 2:3]
        return translation

    @staticmethod
    def data_description(filepath):
        dataframe = pd.read_csv(filepath, delimiter=' ', header=None)
        print(f'Total Number of Rows = {dataframe.shape[0]}')
        print(f'Total Number of Columns = {dataframe.shape[1]}')
        print(f'First 3 Rows \n')
        return dataframe.head(3)

    @staticmethod
    def visualisation(filepath):
        dataframe = pd.read_csv(filepath, delimiter=' ', header=None)
        size = len(dataframe)
        ground_truth = np.zeros((size, 3, 4))
        for i in range(size):
            ground_truth[i] = np.array(dataframe.iloc[i]).reshape((3, 4))
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(ground_truth[:, :, 3][:, 0], ground_truth[:,
                :, 3][:, 1], ground_truth[:, :, 3][:, 2])
        ax.set_xlabel("Movement in X Direction")
        ax.set_ylabel("Movement in Y Direction")
        ax.set_zlabel("Movement in Z Direction")
        ax.set_title("Ground Truth")
        plt.show()
