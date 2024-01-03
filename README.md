# Excalibur

[![Documentation Status](https://readthedocs.org/projects/excalibur-mrm/badge/?version=latest)](https://excalibur-mrm.readthedocs.io/en/latest/?badge=latest)  
[![Tests](https://github.com/uulm-mrm/excalibur/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/uulm-mrm/excalibur/actions/workflows/tests.yml)
[![Codecov](https://codecov.io/gh/uulm-mrm/excalibur/graph/badge.svg?token=MPQQ1SFVNP)](https://codecov.io/gh/uulm-mrm/excalibur)

An open-source Python library for **ex**trinsic sensor **calib**ration.
We provide various solving and optimization methods for different calibration formulations:

 * Point set registration / point-to-point matching
 * Pose set registration / frame-to-frame matching
 * Point-to-line matching
 * Point-to-plane matching
 * Hand-eye calibration
 * Hand-eye robot-world calibration


## Documentation

For installation instructions and the basic usage, see the documentation at [excalibur-mrm.readthedocs.io](https://excalibur-mrm.readthedocs.io).

Furthermore, the `doc` directory provides a detailed readme on how to create the documentation directly from the repository.


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

    DOI: 10.1109/IV55152.2023.10186703  
    IEEE Xplore: https://ieeexplore.ieee.org/document/10186703  
    ArXiv: https://arxiv.org/abs/2305.01407

  * **User Feedback and Sample Weighting for Ill-Conditioned Hand-Eye Calibration**  
    M. Horn, T. Wodtko, M. Buchholz and K. Dietmayer  
    2023 IEEE International Conference on Intelligent Transportation Systems (ITSC)  

    ArXiv: https://arxiv.org/abs/2308.06045

Be aware that we have made improvements and bugfixes since the publications, so the results and run times might differ from the ones in the publications.

You can cite Excalibur directly as

```
@generic{excalibur,
  author = {Markus Horn and Thomas Wodtko},
  title  = {Excalibur: An open-source Python library for extrinsic sensor calibration},
  year   = {2023},
  url    = {https://github.com/uulm-mrm/excalibur},
}
```

Please also cite the respective publication if you are using Excalibur for your own research.
You can find all Bibtex citations in the `CITATIONS.bib` file.
