# Stereo Visual Odometry

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













> To visualise the groud truth pose of the camera, disparity map, and stereo depth map, run the following commands

```
$ cd script/
$ python3 main.py
```

### You will get a 3D output like this

![](https://github.com/NafBZ/Stereo-Visual-Odometry/blob/master/figures/gT.gif)

### Disparity Map

![](https://github.com/NafBZ/Stereo-Visual-Odometry/blob/nafees/figures/disparity.png)

### Stereo Depth Map

![](https://github.com/NafBZ/Stereo-Visual-Odometry/blob/nafees/figures/depth.png)

### Feature Matching Using ORB

![](https://github.com/NafBZ/Stereo-Visual-Odometry/blob/nafees/figures/orb_match.png)

### Feature Matching Using SIFT

![](https://github.com/NafBZ/Stereo-Visual-Odometry/blob/nafees/figures/sift_match.png)

### Contributors:

- [Nafees Bin Zaman](https://github.com/NafBZ)
- [Bruk Gebregziabher](https://github.com/brukg)
- [Jad Mansour](https://github.com/j4dooooo)

## The repository is under development.
