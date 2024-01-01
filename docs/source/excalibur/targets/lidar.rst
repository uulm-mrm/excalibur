.. _Lidar Target Detection:

Lidar Target Detection
======================

The targets described on this page can be detected using ``scripts/detection/lidar_target.py``.
However, the ground plane is detected using ``scripts/detection/lidar_plane.py``.
The configuration classes listed below are provided to the detection scripts as *yaml* files.
Examples are located in the ``scripts/detection/configs`` directory.

.. list-table:: Target types for ``lidar_target.py``.
   :header-rows: 1

   * - Type
     - Config
   * - ``BOARD``
     - :py:class:`LidarBoardConfig <excalibur.targets.lidar.board.LidarBoardConfig>`
   * - ``SPHERE``
     - :py:class:`LidarSphereConfig <excalibur.targets.lidar.sphere.LidarSphereConfig>`

The point clouds are provided as numpy files (*npy*) in a directory.
Their filenames represent the timestamp in nanoseconds as integer.

For most detection algorithms, the point clouds must be structured; otherwise, an error is thrown.
This means that they must have the shape ``(rows, cols, dim)``, where ``rows`` is usually the number of layers and ``cols`` the number of points per layer.
The dimension must be at least 3 and the first three values represent the :math:`x`, :math:`y`, and :math:`z` coordinates.
Invalid measurements can be represented by ``[0.0, 0.0, 0.0]``.
Example files, generated using the `CARLA Simulator <https://carla.org/>`_, are located in the ``test/detection`` directory.

Alternatively, a ROS2 bag can be used as input.
For this, the topic containing ``sensor_msgs/PointCloud2`` messages must be provided.


.. _Ground Plane:

Ground Plane
------------

Detect the ground plane.

.. autoclass:: excalibur.targets.lidar.plane.LidarPlaneConfig
    :member-order: bysource
    :members:
    :undoc-members:


.. _Calibration Board:

Calibration Board
-----------------

Detect and estimate the 3D pose of a specific lidar calibration board.
The rectangular board must have at least three asymmetrically placed holes that enable a unique pose estimation.

.. autoclass:: excalibur.targets.lidar.board.LidarBoardConfig
    :member-order: bysource
    :members:
    :undoc-members:


.. _Sphere:

Sphere
------

Detect and estimate the center position of a sphere.
The exported poses all have an identity rotation since the orientation of a sphere cannot be determined.

.. autoclass:: excalibur.targets.lidar.sphere.LidarSphereConfig
    :member-order: bysource
    :members:
    :undoc-members:
