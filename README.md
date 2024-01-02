# Excalibur

An open-source Python library for **ex**trinsic sensor **calib**ration.
We provide various solving and optimization methods for different calibration formulations:

 * Point set registration / point-to-point matching
 * Pose set registration / frame-to-frame matching
 * Point-to-line matching
 * Point-to-plane matching
 * Hand-eye calibration
 * Hand-eye robot-world calibration


## Installation

Clone the repository and run the following command in your Python environment:

```bash
python3 -m pip install .
```

This command should automatically install Excalibur and all requirements.
If the installation fails, first make sure that pip is updated to the latest version.

Supported Python versions: `3.8`, `3.9`, `3.10`


## Documentation

The `doc` directory provides a detailed readme on how to create the documentation directly from the repository.


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
Please cite the respective publication if you are using Excalibur for your own research.
You can find the Bibtex citations in the `CITATIONS.bib` file.
