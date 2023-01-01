import cv2
import datetime
import numpy as np


def disparity_mapping(left_image, right_image, rgb=False, verbose=False):
    '''
    Takes a stereo pair of images from the sequence and
    computes the disparity map for the left image.

    :params left_image: image from left camera
    :params right_image: image from right camera

    '''

    if rgb:
        num_channels = 3
    else:
        num_channels = 1

    # Empirical values collected from a tutorial
    sad_window = 6  # sad = sum of absolute differences
    num_disparities = sad_window*16
    block_size = 11

    # Using SGBM matcher(Hirschmuller algorithm)
    matcher = cv2.StereoSGBM_create(numDisparities=num_disparities,
                                    minDisparity=0,
                                    blockSize=block_size,
                                    P1=8 * num_channels * sad_window ** 2,
                                    P2=32 * num_channels * sad_window ** 2,
                                    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
                                    )
    if rgb:
        left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

    # Computation time to get disparity map
    start = datetime.datetime.now()
    left_image_disparity_map = matcher.compute(
        left_image, right_image).astype(np.float32)/16
    end = datetime.datetime.now()

    # In case of check the computation time
    if verbose:
        print(
            f'Time to compute disparity map using Stereo SGBM:', end-start)

    return left_image_disparity_map


# Decompose camera projection Matrix
def decomposition(p):
    '''
    :params p: camera projection matrix

    '''
    # Decomposing the projection matrix
    intrinsic_matrix, rotation_matrix, translation_vector, _, _, _, _ = cv2.decomposeProjectionMatrix(
        p)

    # Scaling and removing the homogenous coordinates
    translation_vector = (translation_vector / translation_vector[3])[:3]

    return intrinsic_matrix, rotation_matrix, translation_vector


# Calculating depth information
def depth_mapping(left_disparity_map, left_intrinsic, left_translation, right_translation, rectified=True):
    '''

    :params left_disparity_map: disparity map of left camera
    :params left_intrinsic: intrinsic matrix for left camera
    :params left_translation: translation vector for left camera
    :params right_translation: translation vector for right camera

    '''
    # Focal length of x axis for left camera
    focal_length = left_intrinsic[0][0]

    # Calculate baseline of stereo pair
    if rectified:
        baseline = right_translation[0] - left_translation[0]
    else:
        baseline = left_translation[0] - right_translation[0]

    # Avoid instability and division by zero
    left_disparity_map[left_disparity_map == 0.0] = 0.1
    left_disparity_map[left_disparity_map == -1.0] = 0.1

    # depth_map = f * b/d
    depth_map = np.ones(left_disparity_map.shape)
    depth_map = (focal_length * baseline) / left_disparity_map

    return depth_map


# Let's make an all-inclusive function to get the depth from an incoming set of stereo images
def stereo_depth(left_image, right_image, P0, P1, rgb=False, verbose=False):
    '''
    Takes stereo pair of images and returns a depth map for the left camera. 

    :params left_image: image from left camera
    :params right_image: image from right camera
    :params P0: Projection matrix for the left camera
    :params P1: Projection matrix for the right camera

    '''
    # First we compute the disparity map
    disp_map = disparity_mapping(left_image,
                                 right_image,
                                 rgb=rgb,
                                 verbose=verbose)

    # Then decompose the left and right camera projection matrices
    l_intrinsic, l_rotation, l_translation = decomposition(
        P0)
    r_intrinsic, r_rotation, r_translation = decomposition(
        P1)

    # Calculate depth map for left camera
    depth = depth_mapping(disp_map, l_intrinsic, l_translation, r_translation)

    return depth, disp_map
