.. _Calibration:

Calibration
===========

Transformation data are stored and passed using our open-source Python library ``motion3d``, which handles, e.g., the conversion between different transformation representations or the conversion between motions and poses.
For more information, visit the ``motion3d`` repository: https://github.com/mhorn11/motion3d

Besides the available ``scripts``, you can also check our unit tests in the ``test`` directory or our evaluation scripts in the ``publications`` directory for more examples.

Please be aware that the interfaces also might change between different version, as we are still trying to improve the usability.
If you have any issues or suggestions, don't hesitate to write an issue or to contact us directly via e-mail.


Formulations
------------

| **Point Set Registration / Point-to-Point matching**
| Register point detections, e.g., checkerboard corners or sphere centers.
| :math:`\bm{a} = X(\bm{b})`

| **Pose Set Registration / Frame-to-Frame matching**
| Register pose detections, e.g., of a ChArUco board or a board with holes for a lidar.
| :math:`A = X \circ B`

| **Point-to-Line Matching**
| Register point detections to 3D lines, e.g., the center of a sphere detected as 3D point in lidar measurements with its center detected in camera images, i.e., a viewing ray/line.
  If the lines stem from camera images, i.e., pixel positions, this formulation is related to *Perceptive-n-Point (PnP)*.

| **Point-to-Plane Matching**
| Register point detections to 3D planes, e.g., checkerboard corners detected in camera images with the board plane detected in lidar measurements.

| **Hand-Eye Calibration**
| Calibrate two sensors based on per-sensor ego-motion measurements.
| :math:`A \circ X = X \circ B`

| **Hand-Eye Robot-World Calibration (HERW)**
| A simular calibration problem to hand-eye calibration, but with two unknown transformations :math:`X` and :math:`Y`.
| :math:`A \circ X = Y \circ B`


Implementation
""""""""""""""

All supported formulations have their own base class that provides a ``Factory`` function, enabling string-based class initialization.
Further, all methods can also be initialized directory using the respective class.
Each class provides a method for setting the input data, e.g., ``set_transformations()`` for hand-eye calibration, and a ``calibrate()`` method for computing the result.
Furthermore, the ``configure()`` method makes it possible to provide additional configuration parameters.
The result contains the estimated calibration, run time, and other auxiliary data.

Furthermore, for most formulations, a corresponding RANSAC (random sample consensus) implementation is provided, which automatically handles the sampling and inlier detection.
The RANSAC methods are also child classes of the respective base class of each formulation.

The following pages provide the base class and all child classes for all formulations:

.. toctree::
   :maxdepth: 1

   ./calibration/point2point
   ./calibration/frame2frame
   ./calibration/point2line
   ./calibration/point2plane
   ./calibration/hand_eye
   ./calibration/herw
   ./calibration/ransac


Measurement Conditions
""""""""""""""""""""""

In order to achieve a unique solution, a minimum number of measurements that fulfill specific conditions is required.
The following table gives an overview of all formulations:

.. list-table::
   :header-rows: 1

   * - Formulation
     - Input
     - Min. Meas.
     - Conditions
   * - Point Set Registration
     - points
     - 3
     - non-colinear
   * - Pose Set Registration
     - poses
     - 1
     -
   * - Point-to-Line Matching
     - points, lines
     - 3
     - non-coplanar points
   * - Point-to-Plane Matching
     - points, planes
     - ?
     - ?
   * - Hand-Eye Calib.
     - motions
     - 2
     - non-parallel rotation axes
   * - HERW Calib.
     - poses
     - 3
     - non-parallel inter-pose rotation axes

Especially the non-parallel rotation axes for hand-eye and hand-eye robot-world calibration can often not be achieved, e.g., when calibrating sensor mounted on vehicles with planar motion.
Hence, a unique solution is not possible in this case without incorporating a priori knowledge.
More details on this issue are given in [:ref:`horn2023user <horn2023user>`].
The next section describes extensions that enable hand-eye and hand-eye robot-world calibration with planar motion data.


Extensions
""""""""""""

| **Hand-Eye Calibration with Known Ground Planes**
| In case the ground plane in both sensors is available, the estimation can be reduced to a 2D transformation.
| More details are given in [:ref:`horn2021online <horn2021online>`].
| Supported methods: :py:class:`DualQuaternionQCQPPlanar <excalibur.calibration.hand_eye.DualQuaternionQCQPPlanar>`

| **Hand-Eye and HERW Calibration with Known Translation Norm**
| If the norm of the translation is known or can be measured, it can be provided as a priori knowledge, enabling a unique solution.
| More details are given in [:ref:`horn2023extrinsic <horn2023extrinsic>`].
| Supported hand-eye methods: :py:class:`DualQuaternionQCQP <excalibur.calibration.hand_eye.DualQuaternionQCQP>`
| Supported HERW methods:
  :py:class:`DualQuaternionQCQPSignSampling <excalibur.calibration.herw.DualQuaternionQCQPSignSampling>`,
  :py:class:`DualQuaternionQCQPSeparableInit <excalibur.calibration.herw.DualQuaternionQCQPSeparableInit>`,
  :py:class:`DualQuaternionQCQPSeparableRANSACInit <excalibur.calibration.herw.DualQuaternionQCQPSeparableRANSACInit>`

| **Hand-Eye Calibration with Unknown Translation Scaling**
| For monocular odometry, the scaling of the estimated translation is not known.
  Hence, the scaling must be estimated in addition to the calibration.
  Here, it is always assumed that sensor *a* is correctly scaled and the scaling of sensor *b* is not known.
| More details are given in [:ref:`wodtko2021globally <wodtko2021globally>`].
| Supported methods:
  :py:class:`DualQuaternionQCQPScaled <excalibur.calibration.hand_eye.DualQuaternionQCQPScaled>`,
  :py:class:`MatrixQCQPScaled <excalibur.calibration.hand_eye.MatrixQCQPScaled>`,
  :py:class:`SchmidtDQ <excalibur.calibration.hand_eye.SchmidtDQ>`,
  :py:class:`SchmidtHM <excalibur.calibration.hand_eye.SchmidtHM>`,
  :py:class:`Wei <excalibur.calibration.hand_eye.Wei>`


Scripts
-------

The ``scripts/calibration`` directory provides scripts for most of the supported calibration formulations.

The inputs are provided as ``motion3d`` files.
The respective calibration method settings are provided as configuration file.
For hand-eye and hand-eye robot-world calibration with known translation norm, the required a priori knowledge is provided using an additional file.
Example configurations and norm file templates are provided in the ``scripts/calibration/configs`` directory.

The following page provides details on the output format:

.. toctree::
   :maxdepth: 1

   ./calibration/calibration_manager
