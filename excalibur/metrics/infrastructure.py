import numpy as np

from excalibur.utils.image import project_image_point_to_plane


def line_projection_errors(pred_y, ground_plane_wc, intrinsics, lines):
    # transform ground plane normal to sensor coordinates
    ground_plane_sc = ground_plane_wc.transform(pred_y.inverse())

    # iterate lines
    line_errors_abs = []
    line_errors_rel = []

    for line in lines:
        # project line points to plane
        point1 = project_image_point_to_plane(line.point1, ground_plane_sc, intrinsics)
        point2 = project_image_point_to_plane(line.point2, ground_plane_sc, intrinsics)

        # calculate projected distance and compare to measured distance
        distance = np.linalg.norm(point1 - point2)
        dist_diff = np.abs(distance - line.length)
        rel_error = dist_diff / line.length
        line_errors_abs.append(dist_diff)
        line_errors_rel.append(rel_error)

    return line_errors_abs, line_errors_rel
