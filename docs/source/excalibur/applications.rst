Applications
============

This page gives calibration examples for pairs of different sensor types.
An overview of all targets and detection methods is given in :ref:`Target Detection`.
An overview of all calibration formulations is given in :ref:`Calibration`.


Camera-Lidar Calibration
------------------------

The camera to lidar calibration can be obtained from detections of a calibration board with holes using point set registration (point-to-point matching) or pose set registration (frame-to-frame matching).
The board must have three asymmetrically placed holes for a unique pose estimation in the lidar and a ChArUco print for a precise pose estimation in camera images.

The board should be placed at different angles and positions within the camera image.
However, large angles or distances can especially cause issues with the ChArUco detection.

| The board is detected based on its holes in the lidar as described in :ref:`Calibration Board`.
| In the camera, it is detected based on the ChArUco print using :ref:`ChArUco Board`.

Finally, either :ref:`Point Set Registration` or :ref:`Pose Set Registration` can be used for calibration.
If the orientation of the board is reliably detected in both sensors, pose set registration achieves better results.
Otherwise, point set registration is the better choice.

To verify the result, you can use ``scripts/visualization/show_camera_lidar.py`` for visualizing the transformed point clouds projected into the camera images.


Lidar-Lidar Calibration
-----------------------

The lidar to lidar calibration can be obtained from sphere detections using point set registration (point-to-point matching).

At least 3 non-colinear measurements are required for a unique solution.
Hence, the sphere should be placed at various positions around the vehicle.
For a good detection, the sphere can be placed approximately at the sensor height, where the layer resolution of most lidars is higher.

The sphere is detected separately in both lidars as described in :ref:`Sphere`, and :ref:`Point Set Registration` is used for calibration.

To verify the result, you can use ``scripts/visualization/show_lidars.py`` for visualizing the measurements transformed into the same coordinate system.
