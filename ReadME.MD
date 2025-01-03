# Stereo Visual Odometry

![GitHub Repo stars](https://img.shields.io/github/stars/NafBZ/Stereo-Visual-Odometry?style=for-the-badge&logo=github&labelColor=black&color=green)
![GitHub forks](https://img.shields.io/github/forks/NafBZ/Stereo-Visual-Odometry?style=for-the-badge&logo=github)
[![Static Badge](https://img.shields.io/badge/python-3.9-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)



This repo contains a basic pipeline to implement stereo visual odometry for road vehicles. For experimental evaluation and validation KITTI dataset has been used. Thanks to [Prof. Dr. Ivan Marković](https://www.fer.unizg.hr/en/ivan.markovic) for his amazing lectures on computer vision topics where we learn the core concepts of feature extractions, feature matching, triangulation, motion estimation etc. Moreover, we find this youtube channel named [
First Principles of Computer Vision](https://www.youtube.com/@firstprinciplesofcomputerv3258) extremely useful.


## Directory Structure

```
├── config
│   └── initial_config.yaml
├── datasets
|   ├── calibration
|   ├── poses
|   └── sequences
├── figures
├── License.MD
├── ReadME.MD
├── requirements.txt
└── script
    ├── dataloader.py
    ├── main.py
    ├── playground.py
    ├── utils.py
    └── visual_odometry.py
```
All the executables are inside the [script](https://github.com/NafBZ/Stereo-Visual-Odometry/tree/master/script) folder. The *dataloader.py* file is responsible for creating a data loader instance to read the images from the dataset according to the executables. The *utils.py* file contains all the implemented methods that we have used to create the visual odometry pipeline. In the *playground.py* file, you can check the outputs of individual methods (implemented in *utils.py*) and how they are working. Due to the size, the **KITTI** dataset is not available in this repo. Thus,

> Download the KITTI odometry dataset. Copy the _sequence_ folder from the KITTI dataset and paste it inside the **datasets** folder.

Inside the [config](https://github.com/NafBZ/Stereo-Visual-Odometry/tree/master/config) folder, you can tweak values for different parameters and change the tracks to evaluate the visual odometry performance.

## Visual Odometry Pipeline
This is an overview of how we have implemented this project. 

<p align="center">
  <img src="https://github.com/NafBZ/Stereo-Visual-Odometry/blob/master/figures/pipeline.png" />
</p>

We start creating a data loader class to retrieve images, ground truth and calibration parameters from the KITTI dataset. Due to low memory (RAM is < 16 GB :disappointed:), we utilize the generator option of python. Instead of reading all the images at once, it reads images sequentially inside a loop. Next we pass the images to a wrapper function named stereo depth. Inside the function, for each itearation a disparity maps is construted from a pair of stereo images. To construct disparity map, we have used the [StereoSGBM](https://docs.opencv.org/3.4/d2/d85/classcv_1_1StereoSGBM.html) function from OpenCV which is an implementation of Hirschumuller Algorithm. A disparity map is depicted below:   
![](https://github.com/NafBZ/Stereo-Visual-Odometry/blob/master/figures/disparity.png)

Using the disparity map, we get the depth of the scene using the formula:

$$ \huge{z = {f \times b \over d}}$$

Where, $z$ is the depth, $f$ is focal length, $b$ is baseline of stereo camera and $d$ is disparity. A depth map is shown underneath.

![](https://github.com/NafBZ/Stereo-Visual-Odometry/blob/master/figures/depth.png)

Now we extract features (Keypoints and Descriptors) using **SIFT** or **ORB** from two sequential frames of the left camera. We pass these descriptors to a function in order to compute matching features between two sequential frames. For that we use [Brute-Force Matching](https://docs.opencv.org/4.x/d3/da1/classcv_1_1BFMatcher.html) function with L2-norm from OpenCV. This outputs a lot of matching points, however, not all of them are correct. Thus, we put a threshold value to filter out weak matches. From our experiment, it turns out, threshold value between 0.25 to 0.45 provide a sufficient number of good matches. The [Distinctive Image Features from Scale-Invariant Keypoints](https://link.springer.com/article/10.1023/B:VISI.0000029664.99615.94) paper is a great source to understand SIFT and the threshold characteristic. Although ORB computes faster, in KITTI dataset for our pipeline SIFT performs better. Examples are given below:
<br></br>
*Detector: SIFT - Threshold: 0.25*
![](https://github.com/NafBZ/Stereo-Visual-Odometry/blob/master/figures/sift_match.png)
<br></br>
*Detector: ORB - Threshold: 0.75*
![](https://github.com/NafBZ/Stereo-Visual-Odometry/blob/master/figures/orb_match.png)


From the above images, we see that, SIFT provides us a lot of feature matching points even considering only 25% of matches. Whereas, ORB provide much less number of matching points. After we get sufficient matching points, we find the 3D points using the keypoints, matches, and camera's intrinsic parameters. In the motion estimation function, the depth points $z$ is retrieved from the given matches. And using the following formula $x, y$ values are retrieved as well. 

$$ \huge{x = {z*(u-cx) \over fx}} $$

$$ \huge{y = {z*(v-cy) \over fy}} $$

Here, $u, v$ are pixel coordinates, $fx, fy$ are focal distance in the $x, y$ direction and $cx, cy$ are the optical center of image. We get the focal distance and the optical center from the camera intrinsic matrix. From these 3D points we estimate the motion of the camera. Next, we run [solvePnPRansac](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html) algorithm to filter out the outliers from the 3D points for better estimation. Finally, we plot the results.

# How to run the project

First of all, clone this repository using the following
```
$ git clone https://github.com/NafBZ/Stereo-Visual-Odometry.git
```
Then,
> Download the KITTI odometry dataset. Copy the _sequence_ folder from the KITTI dataset and paste it inside the **datasets** folder.

Once you have cloned this repository, download the dataset and placed inside the right directory, go to the script directory and run the *main.py* file.

```
$ cd script/
$ python3 main.py
```
Inside the script directory, you can play with different functions and see the output by uncommenting necessary lines and run the *playground.py* file.
```
$ python3 playground.py
```

## Performance Evaluation through Plotting Trajectory

> Lets plot the Ground Truth First

<p align="center">
  <img src="https://github.com/NafBZ/Stereo-Visual-Odometry/blob/master/figures/gT.gif" />
</p>


> Now, lets see if our implemented algorithm can estimate motion from stereo images or not

<p align="center">
  <img src="https://github.com/NafBZ/Stereo-Visual-Odometry/blob/master/figures/Trajectory.gif" />
</p>

> Hurray :v:, it works :clap: Now it's the time to reveal the actual results.



Sequence 00                |  Sequence 01
:-------------------------:|:-------------------------:
![](https://github.com/NafBZ/Stereo-Visual-Odometry/blob/master/figures/estim_00.png)  |  ![](https://github.com/NafBZ/Stereo-Visual-Odometry/blob/master/figures/estim_01.png)

From stereo images, we can reliably infer motion thanks to our established visual odometry workflow. In this case, though, a more precise estimate is needed. A large quantity of drift occurs when the errors are accumulated. Noisy feature matches is another reason to increase the error. A more precise and error-free estimate can be derived from combining lidar data with other sources. Also the [Kalman filter](https://en.wikipedia.org/wiki/Kalman_filter) allows us to simultaneously predict and update the motion which overall provides better trajectory with less error. Many methods exist, including bundle adjustment, that can be used to refine the predicted course. Feel free to play with parameters and tracks.

## Performance Evaluation using SLAM Toolbox

To measure the performance of the algorithm's estimated trajectory, an open source SLAM toolbox named [evo](https://github.com/MichaelGrupp/evo) has been used. Please check their github repository for extensive installation purposes (including virtual environment). You can install evo using pypi. Just run the follwing command in your terminal.

```
$ pip install evo --upgrade --no-binary evo
```

To evaluate the trajectory you can simply run the following code. Make sure you put the correct file name in order to visualise.

```
$ cd datasets/poses/
$ evo_traj test_03.txt --ref=03.txt -p --plot_mode=xz
```

<p align="center">
  <strong>Trajectory</strong> of Sequence 3 
</p>

<p align="center">
  <img src="https://github.com/NafBZ/Stereo-Visual-Odometry/blob/master/figures/seq3_plot3.png" />
</p>

<p align="center">
  <strong>Absolute Pose Error</strong> in X,Y,Z Axis - Sequence 3
</p>

<p align="center">
  <img src="https://github.com/NafBZ/Stereo-Visual-Odometry/blob/master/figures/seq3_xyz.png" />
</p>

<p align="center">
  <strong>Absolute Pose Error</strong> in Roll, Pitch, Yaw - Sequence 3
</p>

<p align="center">
  <img src="https://github.com/NafBZ/Stereo-Visual-Odometry/blob/master/figures/seq3_rpy.png" />
</p>


As can be seen in Sequence 3, our implemented visual odometry functions admirably, with an APE (Absolute Positioning Error) of less than 1.6 over the course of the trajectory's greatest extent. Reasons for this include the ease of the track and the superior feature matching offered by stereo images. The y-axis is somewhat off, but the x- and z-axis inaccuracies are negligible.


> Several other matrices have been plotted to understand the performance of the algorithm. Some of them are given below.

You can run the following code and save the result in a zip file.
```
$ cd datasets/poses/
$ evo_ape 03.txt test_03.txt -va --plot --plot_mode xz --save_results results.zip
```

<p align="center">
  <strong>RMSE</strong> 
</p>


<p align="center">
  <img src="https://github.com/NafBZ/Stereo-Visual-Odometry/blob/master/figures/seq3_plot2.png" />
</p>

<br></br>
For statistical plotting just run the following command with the zip file that you saved
```
$ evo_res results.zip -p
```

Violin Plot - APE          |  Density Plot - APE
:-------------------------:|:-------------------------:
![](https://github.com/NafBZ/Stereo-Visual-Odometry/blob/master/figures/new5.png)  |  ![](https://github.com/NafBZ/Stereo-Visual-Odometry/blob/master/figures/new3.png)

**This is being done as part of a seminar for the Robotic Sensing, Perception, and Actuation course at FER**


<p align="center">
  <strong>Thanks for Visiting. Happy Coding</strong> :computer:
</p>
