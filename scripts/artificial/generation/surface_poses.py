#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Optional, Union

from mayavi import mlab
import motion3d as m3d
import numpy as np
import yaml

from excalibur.utils.gauss import GaussConfig, multi_gauss_2d, multi_gauss_grad_2d


def main():
    parser = argparse.ArgumentParser(description="Generate poses on a surface.")
    parser.add_argument('--output', type=str, default=None,
                        help="Output directory")
    parser.add_argument('--max-samples', type=int, default=None,
                        help="Maximum number of samples")
    parser.add_argument('--multi-scale', action='store_true',
                        help="Generate multiple height scalings")
    parser.add_argument('--axes', action='store_true',
                        help="Visualize axes")
    args = parser.parse_args()

    run(output=args.output, max_samples=args.max_samples, multi_scale=args.multi_scale, show_axes=args.axes)


def run(output: Optional[Union[Path, str]] = None, max_samples: Optional[int] = None, multi_scale: bool = False,
        show_axes: bool = False):
    # process arguments
    output_dir = Path(output) if output is not None else None

    # 2d config
    radius = 2.0
    circle_factor = 2
    linear_factor = 0.75
    t_max = 4 * np.pi * radius
    n_samples = 100

    if max_samples is None:
        max_samples = n_samples

    # surface config
    gauss_cfgs = [
        GaussConfig(mean=[1.0, 2.0], cov=10.0, height=4.0),
        GaussConfig(mean=[-2.0, 15.0], cov=20.0, height=3.0),
    ]
    if multi_scale:
        scale_factors = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,
                         0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    else:
        scale_factors = [1.0]

    # plot config
    surface_step = 0.1
    surface_margin = 1.0

    # iterate scaling factors
    for scale in scale_factors:
        # print
        print(f"Generate surface for scale {scale:.1f}")

        # 2d motion
        t_vals = np.linspace(0, t_max, n_samples)
        pos_2d = np.column_stack((
            circle_factor * radius * np.cos(t_vals / radius),
            radius * np.sin(circle_factor * t_vals / radius) + t_vals * linear_factor
        ))

        # 2d gradient
        grad_2d = np.column_stack((
            -circle_factor * np.sin(t_vals / radius),
            circle_factor * np.cos(circle_factor * t_vals / radius) + linear_factor
        ))

        # z position (height)
        pos_z = scale * multi_gauss_2d(pos_2d, gauss_cfgs)

        # surface gradient
        surf_grad = scale * multi_gauss_grad_2d(pos_2d, gauss_cfgs)

        # surface normal (z-axis of frame)
        surf_normal = np.zeros((n_samples, 3))
        surf_normal[:, 0] = surf_grad[0, :]
        surf_normal[:, 1] = surf_grad[1, :]
        surf_normal[:, 2] = 1.0
        surf_normal /= np.linalg.norm(surf_normal, axis=1, keepdims=True)

        # x and y direction in 3d (x-axis and y-axis of frame)
        dir_x_3d_flat = np.zeros((n_samples, 3))
        dir_x_3d_flat[:, 0] = grad_2d[:, 0]
        dir_x_3d_flat[:, 1] = grad_2d[:, 1]
        dir_x_3d_flat[:, 2] = 0
        dir_x_3d_flat /= np.linalg.norm(dir_x_3d_flat, axis=1, keepdims=True)

        dir_y_3d = np.cross(surf_normal, dir_x_3d_flat)
        dir_y_3d /= np.linalg.norm(dir_y_3d, axis=1, keepdims=True)
        dir_x_3d = np.cross(dir_y_3d, surf_normal)
        dir_x_3d /= np.linalg.norm(dir_x_3d, axis=1, keepdims=True)

        # storage
        if output_dir is not None:
            # create transforms and append to container
            data = m3d.TransformContainer(has_stamps=False, has_poses=True)
            for i in range(max_samples):
                t_vec = np.array([pos_2d[i, 0], pos_2d[i, 1], pos_z[i]])
                R_mat = np.column_stack((dir_x_3d[i, :], dir_y_3d[i, :], surf_normal[i, :]))
                transform = m3d.MatrixTransform(t_vec, R_mat).normalized_()
                data.append(transform)

            # create output
            output_subdir = output_dir / f'scale_{scale:.3f}'
            output_subdir.mkdir(exist_ok=True, parents=True)
            output_path = output_subdir / 'poses.m3d'

            # store data
            motion_data = m3d.MotionData(m3d.TransformType.kMatrix, data)
            m3d.M3DWriter.write(str(output_path), motion_data, m3d.M3DFileType.kBinary)

            # store config
            config_data = {
                'scale': float(scale),
            }
            config_path = output_subdir / 'config.yaml'
            with open(config_path, 'w') as file:
                yaml.dump(config_data, file)

        # visualization
        else:
            # sample surface
            surf_x = np.arange(np.min(pos_2d[:max_samples, 0]) - surface_margin,
                               np.max(pos_2d[:max_samples, 0]) + surface_margin, surface_step)
            surf_y = np.arange(np.min(pos_2d[:max_samples, 1]) - surface_margin,
                               np.max(pos_2d[:max_samples, 1]) + surface_margin, surface_step)
            surf_xx, surf_yy = np.meshgrid(surf_x, surf_y)
            surf_2d = np.column_stack((surf_xx.reshape(surf_xx.shape[0] * surf_xx.shape[1]),
                                       surf_yy.reshape(surf_yy.shape[0] * surf_yy.shape[1])))
            surf_z = scale * multi_gauss_2d(surf_2d, gauss_cfgs)
            surf_zz = surf_z.reshape(surf_xx.shape)

            # open figure
            mlab.figure(bgcolor=(1, 1, 1))
            mlab.surf(surf_xx.T, surf_yy.T, surf_zz.T, colormap='gray', vmin=-2.0, vmax=4.0)
            mlab.contour_surf(surf_xx.T, surf_yy.T, surf_zz.T + 0.005,
                              color=(0.3, 0.3, 0.3), line_width=4.0)
            mlab.plot3d(pos_2d[:max_samples, 0], pos_2d[:max_samples, 1], pos_z[:max_samples],
                        color=(31 / 255, 119 / 255, 180 / 255), tube_radius=0.07, tube_sides=18)

            if show_axes:
                mlab.quiver3d(pos_2d[:max_samples, 0], pos_2d[:max_samples, 1], pos_z[:max_samples],
                              dir_x_3d[:max_samples, 0], dir_x_3d[:max_samples, 1], dir_x_3d[:max_samples, 2],
                              color=(1, 0, 0))
                mlab.quiver3d(pos_2d[:max_samples, 0], pos_2d[:max_samples, 1], pos_z[:max_samples],
                              dir_y_3d[:max_samples, 0], dir_y_3d[:max_samples, 1], dir_y_3d[:max_samples, 2],
                              color=(0, 1, 0))
                mlab.quiver3d(pos_2d[:max_samples, 0], pos_2d[:max_samples, 1], pos_z[:max_samples],
                              surf_normal[:max_samples, 0], surf_normal[:max_samples, 1], surf_normal[:max_samples, 2],
                              color=(0, 0, 1))
            mlab.view(azimuth=190, distance=30, elevation=40)
            mlab.show()


if __name__ == '__main__':
    main()
