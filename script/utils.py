import cv2
import numpy as np
import matplotlib.pyplot as plt
import yaml


# Load Config File
with open("../config/initial_config.yaml", "r") as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as error:
        print(error)

# Declare Necessary Variables
rgb_value = config['parameters']['rgb']
rectified_value = config['parameters']['rectified']
detector_name = config['parameters']['detector']
max_depth_value = config['parameters']['max_depth']


############################################ Stereo Depth Estimation #########################################

def disparity_mapping(left_image, right_image, rgb=rgb_value):
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

    # Empirical values collected from a OpenCV website
    num_disparities = 6*16
    block_size = 7

    # Using SGBM matcher(Hirschmuller algorithm)
    matcher = cv2.StereoSGBM_create(numDisparities=num_disparities,
                                    minDisparity=0,
                                    blockSize=block_size,
                                    P1=8 * num_channels * block_size ** 2,
                                    P2=32 * num_channels * block_size ** 2,
                                    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
                                    )
    if rgb:
        left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

    # Disparity map
    left_image_disparity_map = matcher.compute(
        left_image, right_image).astype(np.float32)/16

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
def depth_mapping(left_disparity_map, left_intrinsic, left_translation, right_translation, rectified=rectified_value):
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


def stereo_depth(left_image, right_image, P0, P1, rgb=rgb_value):
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
                                 rgb=rgb)

    # Then decompose the left and right camera projection matrices
    l_intrinsic, l_rotation, l_translation = decomposition(
        P0)
    r_intrinsic, r_rotation, r_translation = decomposition(
        P1)

    # Calculate depth map for left camera
    depth = depth_mapping(disp_map, l_intrinsic, l_translation, r_translation)

    return depth


############################################ Stereo Depth Estimation #########################################


######################################### Feature Extraction and Matching ####################################

def feature_extractor(image, detector=detector_name, mask=None):
    """
    provide keypoints and descriptors

    :params image: image from the dataset

    """
    if detector == 'sift':
        create_detector = cv2.SIFT_create()
    elif detector == 'orb':
        create_detector = cv2.ORB_create()

    keypoints, descriptors = create_detector.detectAndCompute(image, mask)

    return keypoints, descriptors


def feature_matching(first_descriptor, second_descriptor, detector=detector_name, k=2,  distance_threshold=1.0):
    """
    Match features between two images

    """

    if detector == 'sift':
        feature_matcher = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck=False)
    elif detector == 'orb':
        feature_matcher = cv2.BFMatcher_create(
            cv2.NORM_L2, crossCheck=False)
    matches = feature_matcher.knnMatch(
        first_descriptor, second_descriptor, k=k)

    # Filtering out the weak features
    filtered_matches = []
    for match1, match2 in matches:
        if match1.distance <= distance_threshold * match2.distance:
            filtered_matches.append(match1)

    return filtered_matches


def visualize_matches(first_image, second_image, keypoint_one, keypoint_two, matches):
    """
    Visualize corresponding matches in two images

    """
    show_matches = cv2.drawMatches(
        first_image, keypoint_one, second_image, keypoint_two, matches, None, flags=2)
    plt.figure(figsize=(15, 5), dpi=100)
    plt.imshow(show_matches)
    plt.show()

######################################### Feature Extraction and Matching ####################################


######################################### Motion Estimation ####################################
def motion_estimation(matches, firstImage_keypoints, secondImage_keypoints, intrinsic_matrix, depth, max_depth=max_depth_value):
    """
    Estimating motion of the left camera from sequential imgaes 

    """
    rotation_matrix = np.eye(3)
    translation_vector = np.zeros((3, 1))

    # Only considering keypoints that are matched for two sequential frames
    image1_points = np.float32(
        [firstImage_keypoints[m.queryIdx].pt for m in matches])
    image2_points = np.float32(
        [secondImage_keypoints[m.trainIdx].pt for m in matches])

    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]

    points_3D = np.zeros((0, 3))
    outliers = []

    # Extract depth information to build 3D positions
    for indices, (u, v) in enumerate(image1_points):
        z = depth[int(v), int(u)]

        # We will not consider depth greater than max_depth
        if z > max_depth:
            outliers.append(indices)
            continue

        # Using z we can find the x,y points in 3D coordinate using the formula
        x = z*(u-cx)/fx
        y = z*(v-cy)/fy

        # Stacking all the 3D (x,y,z) points
        points_3D = np.vstack([points_3D, np.array([x, y, z])])

    # Deleting the false depth points
    image1_points = np.delete(image1_points, outliers, 0)
    image2_points = np.delete(image2_points, outliers, 0)

    # Apply Ransac Algorithm to remove outliers
    _, rvec, translation_vector, _ = cv2.solvePnPRansac(
        points_3D, image2_points, intrinsic_matrix, None)

    rotation_matrix = cv2.Rodrigues(rvec)[0]

    return rotation_matrix, translation_vector, image1_points, image2_points

######################################### Motion Estimation ####################################


######################################## Error Calculation ###################################
def root_mean_squared_error(ground_truth, estimated_trajectory):

    num_frames_trajectory = estimated_trajectory.shape[0] - 1

    squared_error = np.sqrt((ground_truth[num_frames_trajectory, 0, 3] - estimated_trajectory[:, 0, 3])**2
                            + (ground_truth[num_frames_trajectory, 1,
                               3] - estimated_trajectory[:, 1, 3])**2
                            + (ground_truth[num_frames_trajectory, 2, 3] - estimated_trajectory[:, 2, 3])**2)**2
    mse = squared_error.mean()
    return np.sqrt(mse)
