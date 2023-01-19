import yaml
from dataloader import DataLoader
from visual_odometry import visual_odometry
from utils import root_mean_squared_error
import numpy as np


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

    # Reset frames to start from the beginning of the image list on a new run. Because we are using generators
    data_handler.reset_frames()

    # Estimated trajectory by our algorithm pipeline
    trajectory = visual_odometry(data_handler)
    trajectory = trajectory.flatten('C')
    with open("../datasets/poses/test_" + str(sequence) + ".txt", "w") as f:
        width = 12
        for k, i in enumerate(trajectory):
            if (k+1) % width == 0:
                f.write(str(i)+"\n")
            else:
                f.write(str(i)+" ")
