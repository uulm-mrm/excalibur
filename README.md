# Excalibur

An open-source Python library for **ex**trinsic sensor **calib**ration.
We provide various solving and optimization methods for different calibration formulations:

 * Point set registration / point-to-point matching
 * Pose set registration / frame-to-frame matching
 * Bundle adjustment / point-to-line matching
 * Hand-eye calibration
 * Hand-eye robot-world calibration


## Installation

Clone the repository and run the following command in your Python environment:

```bash
python3 -m pip install .
```

This command should automatically install Excalibur and all requirements.



## Usage

Transformation data are stored and passed using our open-source Python library `motion3d` which handles, e.g., the conversion between different transformation representations or the conversion between motions and poses.
For more information, please visit the `motion3d` repository: https://github.com/mhorn11/motion3d

All supported formulations have their own base class that provides a `Factory` function, enabling string-based initialization of methods.
Further, all methods can also be initialized directory using the respective class.
Each class provides a method for setting the input data, e.g., `set_transformations(...)` for hand-eye calibration, and a `calibrate` method for computing the result.
The result contains the estimated calibration, run time and other auxiliary data.

We are still working on a detailled documentation.
For now, you can check our unit tests in the `test` directory or our evaluation scripts in the `publications` directory for examples.

Please be aware that the interfaces also might change between different version, as we are still trying to improve the usability.
If you have any issues or suggestions, don't hesitate to write an issue or to contact us directly via e-mail.



## Publications

The `publications` directory contains scripts and data related to the following publications of our `DualQuaternionQCQP` methods for hand-eye calibration and hand-eye robot-world calibration:

  * **Online Extrinsic Calibration Based on Per-Sensor Ego-Motion Using Dual Quaternions**  
    M. Horn, T. Wodtko, M. Buchholz and K. Dietmayer  
    IEEE Robotics and Automation Letters (RA-L) (Volume: 6, Issue: 2, April 2021)

    DOI: 10.1109/LRA.2021.3056352  
    IEEE Xplore: https://ieeexplore.ieee.org/document/9345480  
    ArXiv: https://arxiv.org/abs/2101.11440

  * **Globally Optimal Multi-Scale Monocular Hand-Eye Calibration Using Dual Quaternions**  
    T. Wodtko, M. Horn, M. Buchholz and K. Dietmayer  
    2021 International Conference on 3D Vision (3DV)

    DOI: 10.1109/3DV53792.2021.00035  
    IEEE Xplore: https://ieeexplore.ieee.org/document/9665837  
    ArXiv: https://arxiv.org/abs/2201.04473

  * **Extrinsic Infrastructure Calibration Using the Hand-Eye Robot-World Formulation**  
    M. Horn, T. Wodtko, M. Buchholz and K. Dietmayer  
    2023 IEEE Intelligent Vehicles Symposium (IV)

    ArXiv: https://arxiv.org/abs/2305.01407

Be aware that we have made improvements and bugfixes since the publications, so the results and run times might differ from the ones in the publications.
Please cite the respective publication if you are using Excalibur for your own research.
You can find the Bibtex citations in the `CITATIONS.bib` file.
