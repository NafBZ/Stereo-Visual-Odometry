# Stereo Visual Odometry

This repo contains an implementation of stereo visual odometry algorithms for road vehicles. For experimental evaluation and validation KITTI dataset has been used.

> First, download the KITTI odometry dataset. Copy the *sequence* folder from the KITTI dataset and paste it inside the **datasets** folder. 

> To visualise the groud truth pose of the camera, disparity map, and stereo depth map, run the following commands

```
$ cd scripts/ 
$ python3 visual_odometry.py
```
### You will get a 3D output like this
![](https://github.com/NafBZ/Stereo-Visual-Odometry/blob/master/figures/gT.gif)

### Disparity Map
![](https://github.com/NafBZ/Stereo-Visual-Odometry/blob/nafees/figures/disparity.png)

### Stereo Depth Map
![](https://github.com/NafBZ/Stereo-Visual-Odometry/blob/nafees/figures/depth.png)

### Contributors: 
- [Nafees Bin Zaman](https://github.com/NafBZ)
- [Bruk Gebregziabher](https://github.com/brukg)
- [Jad Mansour](https://github.com/j4dooooo)

## The repository is under development.
