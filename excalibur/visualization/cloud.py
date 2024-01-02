import matplotlib as mpl
import motion3d as m3d
import numpy as np
import open3d as o3d


def crop_cloud(cloud: np.ndarray, min_range: float = 0.0, max_range: float = np.inf) -> np.ndarray:
    cloud_range = np.linalg.norm(cloud[..., :3], axis=-1)
    cloud_indices = (min_range <= cloud_range) & (cloud_range <= max_range)
    return cloud[cloud_indices, ...]


def cloud_to_open3d(cloud, colors=None, pcd=None, cmap='hsv'):
    # check recarray
    if not isinstance(cloud, np.ndarray):
        raise TypeError("Only numpy arrays are supported.")
    if cloud.dtype.names is not None:
        raise TypeError("Recarrays are not supported.")

    # reshape structured cloud and associated colors
    if cloud.ndim == 3:
        cloud = cloud.reshape(cloud.shape[0] * cloud.shape[1], cloud.shape[2])
        if colors is not None and isinstance(colors, np.ndarray):
            if colors.ndim == 3:
                colors = colors.reshape(colors.shape[0] * colors.shape[1], colors.shape[2])
            elif colors.ndim == 2:
                colors = colors.reshape(colors.shape[0] * colors.shape[1])

    # adjust cloud point dimension
    if cloud.shape[1] > 3:
        cloud = cloud[:, :3]
    elif cloud.shape[1] < 3:
        cloud = np.column_stack((cloud, np.zeros((cloud.shape[0], 3 - cloud.shape[1]))))

    # adjust colors
    if colors is not None:
        if isinstance(colors, tuple):
            colors = np.tile(colors, (cloud.shape[0], 1))
        elif isinstance(colors, np.ndarray) and colors.ndim == 1:
            # normalize
            colors_min = np.min(colors)
            colors_max = np.max(colors)
            colors_norm = (colors - colors_min) / (colors_max - colors_min)

            # apply colormap
            cmap = mpl.colormaps.get_cmap(cmap)
            colors = np.array([cmap(c)[:3] for c in colors_norm])

    # open3d cloud
    if pcd is None:
        pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def pose_to_open3d(pose):
    mat = pose.asType(m3d.TransformType.kMatrix).getMatrix()
    cf = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=mat[:3, 3])
    cf.rotate(mat[:3, :3], center=mat[:3, 3])
    return cf


def visualize_cloud(cloud, colors=None, pose=None, cmap='hsv'):
    # cloud
    geometries = [cloud_to_open3d(cloud, colors, cmap=cmap)]

    # pose frames
    if pose is not None:
        if not isinstance(pose, list):
            pose = [pose]
        geometries.extend([pose_to_open3d(p) for p in pose])

    # initialize viewer
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # add geometries
    for g in geometries:
        vis.add_geometry(g)

    # show and clean up
    vis.run()
    vis.destroy_window()
