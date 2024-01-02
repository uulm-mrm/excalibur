.. _Target Detection:

Target Detection
================

Excalibur provides detection scripts for camera and lidar targets in the ``scripts/detection`` directory.
Usually, the measurements, i.e., images or point clouds, are provided as individual files within an input directory.

For reading data from ROS2 bags, ROS2 must be installed and sourced.
This is automatically verified when importing the :py:mod:`excalibur.ros` module.

An overview of different targets is given on the following pages:

.. toctree::
   :maxdepth: 1

   ./targets/camera
   ./targets/lidar
