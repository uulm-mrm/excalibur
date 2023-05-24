# Publications


## Online Extrinsic Calibration based on Per-Sensor Ego-Motion Using Dual Quaternions

M. Horn, T. Wodtko, M. Buchholz and K. Dietmayer  
IEEE Robotics and Automation Letters (RA-L) (Volume: 6, Issue: 2, April 2021)

In this work, we propose an approach for extrinsic sensor calibration from per-sensor ego-motion estimates. Our problem formulation is based on dual quaternions, enabling two different online capable solving approaches. We provide a certifiable globally optimal and a fast local approach along with a method to verify the globality of the local approach. Additionally, means for integrating previous knowledge, for example, a common ground plane for planar sensor motion, are described. Our algorithms are evaluated on simulated data and on a publicly available dataset containing RGB-D camera images. Further, our online calibration approach is tested on the KITTI odometry dataset, which provides data of a lidar and two stereo camera systems mounted on a vehicle. Our evaluation confirms the short run time, state-of-the-art accuracy, as well as online capability of our approach while retaining the global optimality of the solution at any time.

DOI: 10.1109/LRA.2021.3056352  
IEEE Xplore: https://ieeexplore.ieee.org/document/9345480  
ArXiv: https://arxiv.org/abs/2101.11440


## Globally Optimal Multi-Scale Monocular Hand-Eye Calibration Using Dual Quaternions

T. Wodtko, M. Horn, M. Buchholz and K. Dietmayer  
2021 International Conference on 3D Vision (3DV)

In this work, we present an approach for monocular hand-eye calibration from per-sensor ego-motion based on dual quaternions. Due to non-metrically scaled translations of monocular odometry, a scaling factor has to be estimated in addition to the rotation and translation calibration. For this, we derive a quadratically constrained quadratic program that allows a combined estimation of all extrinsic calibration parameters. Using dual quaternions leads to low run-times due to their compact representation. Our problem formulation further allows to estimate multiple scalings simultaneously for different sequences of the same sensor setup. Based on our problem formulation, we derive both, a fast local and a globally optimal solving approach. Finally, our algorithms are evaluated and compared to state-of-the-art approaches on simulated and real-world data, e.g., the EuRoC MAV dataset.

DOI: 10.1109/3DV53792.2021.00035  
IEEE Xplore: https://ieeexplore.ieee.org/document/9665837  
ArXiv: https://arxiv.org/abs/2201.04473


## Extrinsic Infrastructure Calibration Using the Hand-Eye Robot-World Formulation

M. Horn, T. Wodtko, M. Buchholz and K. Dietmayer  
2023 IEEE Intelligent Vehicles Symposium (IV)

We propose a certifiably globally optimal approach for solving the hand-eye robot-world problem supporting multiple sensors and targets at once. Further, we leverage this formulation for estimating a geo-referenced calibration of infrastructure sensors. Since vehicle motion recorded by infrastructure sensors is mostly planar, obtaining a unique solution for the respective hand-eye robot-world problem is unfeasible without incorporating additional knowledge. Hence, we extend our proposed method to include a-priori knowledge, i.e., the translation norm of calibration targets, to yield a unique solution. Our approach achieves state-of-the-art results on simulated and real-world data. Especially on real-world intersection data, our approach utilizing the translation norm is the only method providing accurate results.

ArXiv: https://arxiv.org/abs/2305.01407
