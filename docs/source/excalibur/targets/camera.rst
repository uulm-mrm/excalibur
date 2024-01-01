.. _Camera Target Detection:

Camera Target Detection
=======================

The targets described on this page can be detected using ``scripts/detection/camera_target.py``.
The configuration classes listed below are provided to the detection scripts as *yaml* files.
Examples are located in the ``scripts/detection/configs`` directory.

.. list-table:: Target types for ``camera_target.py``.
   :header-rows: 1

   * - Type
     - Config
   * - ``CHECKERBOARD``
     - :py:class:`CheckerboardConfig <excalibur.targets.camera.checkerboard.CheckerboardConfig>`
   * - ``CHARUCO``
     - :py:class:`CharucoBoardConfig <excalibur.targets.camera.charuco.CharucoBoardConfig>`
   * - ``CB_COMBI``
     - :py:class:`CheckerboardCombiConfig <excalibur.targets.camera.checkerboard_combi.CheckerboardCombiConfig>`

The camera images are provided as image files (e.g., *png* or *jpg*) in a directory.
Their filenames represent the timestamp in nanoseconds as integer.
The directory must further contain an ``intrinsics.yaml`` file, providing the camera intrinsics:

.. autoclass:: excalibur.io.camera.CameraIntrinsics
    :members:

Example images, generated using the `CARLA Simulator <https://carla.org/>`_, are located in the ``test/detection`` directory.

Alternatively, a ROS2 bag can be used as input.
For this, the topic containing ``sensor_msgs/Image`` messages must be provided.
The intrinsics are automatically extracted from the respective ``sensor_msgs/CameraInfo`` topic.


.. _Checkerboard:

Checkerboard
------------

Detect and estimate the 3D pose of a checkerboard.

.. note::

   A unique estimation of the 3D pose of a checkerboard from camera images is not always possible.
   This issue, called flip ambiguity, is explained in detail in `IPPE by Toby Collins <https://github.com/tobycollins/IPPE>`_.
   The Checkerboard-ArUco combination can resolve this issue by including additional non-coplanar ArUco markers.

.. autoclass:: excalibur.targets.camera.checkerboard.CheckerboardConfig
    :member-order: bysource
    :members:
    :undoc-members:


.. _ChArUco Board:

ChArUco Board
-------------

Detect and estimate the 3D pose of a ChArUco board.

.. note::

   Since the checkerboard and all ArUco markers are coplanar, the previously mentioned flip-ambiguity can also apply here.

.. autoclass:: excalibur.targets.camera.charuco.CharucoBoardConfig
    :member-order: bysource
    :members:
    :undoc-members:


.. _Checkerboard-ArUco Combination:

Checkerboard-ArUco Combination
------------------------------

Detect and estimate the 3D pose of a checkerboard, supported by ArUco markers with a known approximate relative pose with respect to the checkerboard.
The ArUco markers help to preselect the checkerboard within the camera image.
Furthermore, a non-coplanar ArUco marker can resolve the flip ambiguity of the checkerboard detection.

.. autoclass:: excalibur.targets.camera.checkerboard_combi.ArucoMarker
    :member-order: bysource
    :members:
    :undoc-members:

.. autoclass:: excalibur.targets.camera.checkerboard_combi.CheckerboardCombiConfig
    :member-order: bysource
    :members:
    :undoc-members:
