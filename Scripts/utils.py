import yaml
import pandas as pd
import numpy as np

with open("../config/initial_config.yaml", "r") as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


path = config['data']['pose_path']


# To know the pose of the camera wrt. the world coordinate
class CameraPoses:

    # Read the pose data from the text file
    cam_pose = pd.read_csv(path, delimiter=' ', header=None)

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
        print(f'First 3 Rows \n{dataframe.head(3)}')
