import yaml
from dataloader import DataLoader
from utils import *


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
    next_left_image = data_handler.second_image_left
    left_camera_matrix = data_handler.P0
    right_camera_matrix = data_handler.P1

    # Visualise Ground Truth
    # print(data_handler.gt_visualisation())

    # # To know matrices of each frame wrt. to the world coordinate
    # print(
    #     f'The translation of that frame wrt. world coordinate \n{data_handler.gt_translation_matrix()}\n')
    # print(
    #     f'The rotation of that frame wrt. world coordinate \n{data_handler.gt_rotation_matrix()}\n')

    # data_handler.reset_frames()

    # depth and disparity maps
    # depth, disp_map = stereo_depth(left_image,
    #                                right_image,
    #                                left_camera_matrix,
    #                                right_camera_matrix)

    # plt.figure(figsize=(14, 8))
    # plt.title('Stereo Depth Mapping')
    # plt.imshow(depth)

    # plt.figure(figsize=(14, 8))
    # plt.title('Disparity Mapping')
    # plt.imshow(disp_map)

    # # Extracting features
    # first_keypoints, first_descriptors = feature_extractor(left_image, 'orb')
    # second_keypoints, second_descriptors = feature_extractor(
    #     next_left_image, 'orb')

    # # Matching without filtering
    # matches = feature_matching(
    #     first_descriptors, second_descriptors, detector='orb')
    # print('Number of matches before filtering:', len(matches))

    # # Filtering the weak features
    # matches = feature_matching(
    #     first_descriptors, second_descriptors, detector='orb', distance_threshold=0.75)
    # print('Number of matches after filtering:', len(matches))

    # visualize_matches(left_image, next_left_image,
    #                   first_keypoints, second_keypoints, matches)
