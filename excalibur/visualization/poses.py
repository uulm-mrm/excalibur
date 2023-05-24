import motion3d as m3d
import numpy as np


def plot_positions(ax, positions, *args, **kwargs):
    min_positions = np.min(positions, axis=0)
    max_positions = np.max(positions, axis=0)
    mid_positions = (max_positions + min_positions) / 2
    max_range = np.max((max_positions - min_positions) / 2)

    ax.scatter3D(positions[:, 0], positions[:, 1], positions[:, 2], *args, **kwargs)
    ax.set_xlim(mid_positions[0] - max_range, mid_positions[0] + max_range)
    ax.set_ylim(mid_positions[1] - max_range, mid_positions[1] + max_range)
    ax.set_zlim(mid_positions[2] - max_range, mid_positions[2] + max_range)


def plot_orientations(ax, positions1, positions2, step, *args, **kwargs):
    for index in range(0, len(positions1), step):
        ax.plot([positions1[index, 0], positions2[index, 0]],
                [positions1[index, 1], positions2[index, 1]],
                [positions1[index, 2], positions2[index, 2]], '-', *args, **kwargs)


def plot_poses(ax, transforms, origin=None, pos_step=1, axes_step=0, axes_length=1.0):
    # convert to poses
    transforms.removeStamps_()
    transforms.normalized_()
    transforms.asType_(m3d.TransformType.kMatrix)
    poses = transforms.asPoses_()

    # transform to origin
    if origin is not None:
        poses.changeFrame_(origin.normalized().inverse())

    # create plot data
    plot_data = {'pos': [], 'ax': [], 'ay': [], 'az': []}
    for p in poses:
        matrix = p.getMatrix()
        plot_data['pos'].append(matrix @ np.array([0.0, 0.0, 0.0, 1.0]))
        if axes_step > 0 and axes_length > 0:
            plot_data['ax'].append(matrix @ np.array([axes_length, 0.0, 0.0, 1.0]))
            plot_data['ay'].append(matrix @ np.array([0.0, axes_length, 0.0, 1.0]))
            plot_data['az'].append(matrix @ np.array([0.0, 0.0, axes_length, 1.0]))
    plot_data = {k: np.array(v) for k, v in plot_data.items()}

    # plot data
    plot_positions(ax, plot_data['pos'][::pos_step])
    if axes_step > 0 and axes_length > 0:
        plot_orientations(ax, plot_data['pos'], plot_data['ax'], step=pos_step * axes_step, c='r')
        plot_orientations(ax, plot_data['pos'], plot_data['ay'], step=pos_step * axes_step, c='g')
        plot_orientations(ax, plot_data['pos'], plot_data['az'], step=pos_step * axes_step, c='b')
