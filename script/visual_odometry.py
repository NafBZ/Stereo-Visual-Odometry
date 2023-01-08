import matplotlib.pyplot as plt
import numpy as np
import yaml
from utils import stereo_depth, decomposition, feature_extractor, feature_matching, motion_estimation


# Load Config File
with open("../config/initial_config.yaml", "r") as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as error:
        print(error)

# Declare Necessary Variables
detector_name = config['parameters']['detector']
subset = config['parameters']['subset']
threshold = config['parameters']['distance_threshold']


def visual_odometry(data_handler, detector=detector_name, mask=None, subset=subset, plot=True):
    '''
    Compute the visual odometry using all the components
    '''

    if subset is not None:
        num_frames = subset
    else:
        num_frames = data_handler.frames

    if plot:
        fig = plt.figure(figsize=(14, 14))
        ax = fig.add_subplot(projection='3d')
        ax.view_init(elev=-20, azim=270)
        xs = data_handler.ground_truth[:, 0, 3]
        ys = data_handler.ground_truth[:, 1, 3]
        zs = data_handler.ground_truth[:, 2, 3]

        # Plotting range for better visualisation
        ax.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs)))
        ax.plot(xs, ys, zs, c='dimgray')
        ax.set_title("Ground Truth vs Estimated Trajectory")

    # Create a homogeneous matrix
    homo_matrix = np.eye(4)
    trajectory = np.zeros((num_frames, 3, 4))
    trajectory[0] = homo_matrix[:3, :]

    # From projection matrix retrieve the left camera's intrinsic matrix
    left_instrinsic_matrix, _, _ = decomposition(data_handler.P0)

    if data_handler.low_memory:
        data_handler.reset_frames()
        next_image = next(data_handler.left_images)

    # Loop to iterate all the frames
    for i in range(num_frames - 1):

        # using generator retrieveing images
        if data_handler.low_memory:
            image_left = next_image
            image_right = next(data_handler.right_images)
            next_image = next(data_handler.left_images)

        # If you set the low memory to False, all your images will be stored in your RAM and you can access like a normal array.
        else:
            image_left = data_handler.left_images[i]
            image_right = data_handler.right_images[i]
            next_image = data_handler.left_images[i+1]

        # Estimating the depth of an image
        depth = stereo_depth(image_left,
                             image_right,
                             P0=data_handler.P0,
                             P1=data_handler.P1)

        # Keypoints and Descriptors of two sequential images of the left camera
        keypoint_left_first, descriptor_left_first = feature_extractor(
            image_left, detector, mask)
        keypoint_left_next, descriptor_left_next = feature_extractor(
            next_image, detector, mask)

        # Use feature (e.g. SIFT or ORB) detector to match features
        matches = feature_matching(descriptor_left_first,
                                   descriptor_left_next,
                                   detector=detector,
                                   distance_threshold=threshold)

        # Estimate motion between sequential images of the left camera
        rotation_matrix, translation_vector, _, _ = motion_estimation(
            matches, keypoint_left_first, keypoint_left_next, left_instrinsic_matrix, depth)

        # Initialise a homogeneous matrix (4X4)
        Transformation_matrix = np.eye(4)

        # Build the Transformation matrix using rotation matrix and translation vector from motion estimation function
        Transformation_matrix[:3, :3] = rotation_matrix
        Transformation_matrix[:3, 3] = translation_vector.T

        # Transformation wrt. world coordinate system
        homo_matrix = homo_matrix.dot(np.linalg.inv(Transformation_matrix))

        # Append the pose of camera in the trajectory array
        trajectory[i+1, :, :] = homo_matrix[:3, :]

        if i % 10 == 0:
            print(f'{i} frames have been computed')

        if i == num_frames - 2:
            print('All frames have been computed')

        if plot:
            xs = trajectory[:i+2, 0, 3]
            ys = trajectory[:i+2, 1, 3]
            zs = trajectory[:i+2, 2, 3]
            plt.plot(xs, ys, zs, c='darkorange')
            plt.pause(1e-32)

    if plot:
        plt.show()

    return trajectory
