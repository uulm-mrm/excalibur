from enum import IntEnum

import numpy as np


class EdgeSegmentId(IntEnum):
    NONE = 0
    LEFT_EDGE = 1
    RIGHT_EDGE = 2
    PLATEAU = 3


def lidar_edge_segmentation(cloud, grad_thresh, min_range=None, max_range=None, range_interpolation_threshold=None,
                            min_plateau_len=None, max_plateau_len=None):
    # cloud shape: [layers, angles, point dim]
    assert cloud.ndim == 3 and cloud.shape[2] >= 3

    # calculate gradients
    xyz = cloud[:, :, :3]
    ranges = np.linalg.norm(xyz, axis=2)  # ranges = cloud[:, :, 3]

    # interpolate missing measurements
    if range_interpolation_threshold is not None and range_interpolation_threshold > 0.0:
        range_left1 = np.roll(ranges, 1, axis=1)
        range_right1 = np.roll(ranges, -1, axis=1)
        left_right_diff = range_left1 - range_right1
        interpolation_indices = (ranges < 1e-6) & (left_right_diff <= range_interpolation_threshold)
        ranges[interpolation_indices] = (range_left1[interpolation_indices] + range_right1[interpolation_indices]) / 2.0

    # calculate gradient for the next two points in case the next point is invalid, i.e., (0, 0, 0)
    range_left1 = np.roll(ranges, 1, axis=1)
    range_left2 = np.roll(ranges, 2, axis=1)
    range_right1 = np.roll(ranges, -1, axis=1)
    range_right2 = np.roll(ranges, -2, axis=1)

    grad_left1 = range_left1 - ranges
    grad_left2 = range_left2 - ranges
    grad_right1 = range_right1 - ranges
    grad_right2 = range_right2 - ranges

    grad_left = grad_left1.copy()
    grad_right = grad_right1.copy()
    range_left1_zero = range_left1 < 1e-6
    range_right1_zero = range_right1 < 1e-6
    grad_left[range_left1_zero] = grad_left2[range_left1_zero]
    grad_right[range_right1_zero] = grad_right2[range_right1_zero]
    range_left2_zero = range_left2 < 1e-6
    range_right2_zero = range_right2 < 1e-6
    grad_left[range_left2_zero & range_left2_zero] = np.inf
    grad_right[range_right2_zero & range_right2_zero] = np.inf

    # check edges
    is_left = (grad_left >= grad_thresh) & (grad_right < grad_thresh) & (ranges > 1e-6)
    is_right = (grad_left < grad_thresh) & (grad_right >= grad_thresh) & (ranges > 1e-6)

    # process layers separately
    segmentation = np.zeros((cloud.shape[0], cloud.shape[1])).astype(np.uint8)
    for layer in range(cloud.shape[0]):
        # iterate angles (over the point cloud edge to cover plateaus located at the cut)
        plateau_indices = None
        for iter_angle in range(cloud.shape[1] * 2):
            # valid angle index
            angle = iter_angle % cloud.shape[1]

            # check range
            if (min_range is not None and ranges[layer, angle] < min_range) or \
                    (max_range is not None and ranges[layer, angle] > max_range):
                plateau_indices = None
                continue

            # check stopping criterion on second rotation
            if iter_angle >= cloud.shape[1] and plateau_indices is None:
                break

            # check left or right edge
            if is_left[layer, angle]:
                # check stopping criterion on second rotation
                if iter_angle >= cloud.shape[1]:
                    break

                # start new potential plateau
                plateau_indices = [angle]

            elif is_right[layer, angle]:
                # store and reset potential plateau
                if plateau_indices is not None:
                    plateau_indices.append(angle)
                    # check plateau length
                    plateau_length = np.linalg.norm(xyz[layer, plateau_indices[0], :] -
                                                    xyz[layer, plateau_indices[-1], :])
                    if min_plateau_len is not None and plateau_length < min_plateau_len:
                        continue
                    if max_plateau_len is not None and plateau_length > max_plateau_len:
                        continue

                    # store plateau
                    segmentation[layer, plateau_indices[0]] = EdgeSegmentId.LEFT_EDGE
                    segmentation[layer, plateau_indices[1:-1]] = EdgeSegmentId.PLATEAU
                    segmentation[layer, plateau_indices[-1]] = EdgeSegmentId.RIGHT_EDGE
                plateau_indices = None

                # check stopping criterion on second rotation
                if iter_angle >= cloud.shape[1]:
                    break

            else:
                # continue potential plateau
                if plateau_indices is not None:
                    plateau_indices.append(angle)

    return segmentation
