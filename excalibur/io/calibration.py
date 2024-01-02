from pathlib import Path
from typing import Any, Dict, Optional, Union

import matplotlib.pyplot as plt
import motion3d as m3d
import numpy as np
import networkx as nx
import yaml

from excalibur.utils.logging import logger


def load_calibration(filename: Union[str, Path]) -> Optional[Dict[Any, m3d.TransformInterface]]:
    # check file
    filename = Path(filename)
    if not filename.exists():
        return None

    # load data
    with open(str(filename), 'r') as stream:
        data = yaml.safe_load(stream)

    # iterate data
    calib = {}
    for k, v in data.items():
        ttype = m3d.TransformType.FromChar(v['type'])
        calib[k] = m3d.TransformInterface.Factory(ttype, v['data'], unsafe=True).normalized_()

    return calib


def _serialize_transform(transform):
    # find transform type
    transform_type = None
    for ttype in m3d.TransformType.__members__.values():
        if transform.isType(ttype):
            transform_type = ttype
            break

    # serialize
    data = {
        'type': transform_type.toChar(),
        'data': transform.toList()
    }
    return data


def _deserialize_transform(data, unsafe=False):
    transform_type = m3d.TransformType.FromChar(data['type'])
    return m3d.TransformInterface.Factory(transform_type, data['data'], unsafe=unsafe).normalized_()


class CalibrationManager:
    def __init__(self):
        self._graph = nx.Graph()
        self._data = dict()

    def clear(self):
        self._graph.clear()
        self._data.clear()

    def items(self):
        return self._data.items()

    def frames(self):
        return [node for node in self._graph.nodes]

    def add(self, topic_from: str, topic_to: str, transformation: m3d.TransformInterface, overwrite: bool = False)\
            -> bool:
        if not isinstance(transformation, m3d.TransformInterface):
            raise RuntimeError("Input 'transformation' is not an 'm3d.TransformInterface'.")

        # prevent cycles
        has_nodes = self._graph.has_node(topic_from) and self._graph.has_node(topic_to)
        if has_nodes and nx.has_path(self._graph, topic_from, topic_to):
            # direct connection
            if overwrite and (topic_from, topic_to) in self._data:
                self._data[topic_from, topic_to] = transformation
                return True
            # direct connection (the other way round)
            elif overwrite and (topic_to, topic_from) in self._data:
                del self._data[topic_to, topic_from]
                self._data[topic_from, topic_to] = transformation
                return True
            # no overwrite or indirect connection
            else:
                return False
        else:
            # add nodes and edge
            if not self._graph.has_node(topic_from):
                self._graph.add_node(topic_from)
            if not self._graph.has_node(topic_to):
                self._graph.add_node(topic_to)
            self._graph.add_edge(topic_from, topic_to)

            # add transformation
            self._data[topic_from, topic_to] = transformation
            return True

    def has(self, topic_from: str, topic_to: str) -> bool:
        return nx.has_path(self._graph, topic_from, topic_to)

    def get(self, topic_from: str, topic_to: str) -> Optional[m3d.TransformInterface]:
        try:
            # shortest path
            path = nx.shortest_path(self._graph, topic_from, topic_to)

            # follow path
            transformation = m3d.MatrixTransform()
            for i in range(len(path) - 1):
                if (path[i], path[i + 1]) in self._data:
                    if transformation is None:
                        transformation = self._data[path[i], path[i + 1]]
                    else:
                        transformation *= self._data[path[i], path[i + 1]]
                else:
                    if transformation is None:
                        transformation = self._data[path[i + 1], path[i]]
                    else:
                        transformation /= self._data[path[i + 1], path[i]]

            # check identity transform
            if transformation is None:
                transformation = m3d.MatrixTransform()

            return transformation

        except nx.NodeNotFound:
            return None
        except nx.NetworkXNoPath:
            return None

    def remove(self, topic_from: str, topic_to: str) -> bool:
        if (topic_from, topic_to) in self._data:
            self._graph.remove_edge(topic_from, topic_to)
            del self._data[topic_from, topic_to]
            return True
        elif (topic_to, topic_from) in self._data:
            self._graph.remove_edge(topic_to, topic_from)
            del self._data[topic_to, topic_from]
            return True
        else:
            return False

    def extend(self, other, overwrite: bool = False):
        for (topic_from, topic_to), transform in other._data.items():
            self.add(topic_from, topic_to, transform, overwrite=overwrite)

    def copy(self):
        new_obj = CalibrationManager()
        new_obj._graph = self._graph.copy(as_view=False)
        new_obj._data = {k: v.copy() for k, v in self._data.items()}
        return new_obj

    @classmethod
    def load(cls, filename: Union[str, Path], unsafe: bool = False):
        manager = cls()

        with open(str(filename), 'r') as file:
            data = yaml.safe_load(file)

        for topic_from, sub_data in data.items():
            # add node
            if not manager._graph.has_node(topic_from):
                manager._graph.add_node(topic_from)

            for topic_to, d in sub_data.items():
                # add node
                if not manager._graph.has_node(topic_to):
                    manager._graph.add_node(topic_to)

                # add edge
                manager._graph.add_edge(topic_from, topic_to)

                # add transformation
                manager._data[topic_from, topic_to] = _deserialize_transform(d, unsafe=unsafe)

        # check for cycles
        if len(nx.cycle_basis(manager._graph)) > 0:
            logger.error("Calibration file contains cycles")
            return None

        return manager

    def save(self, filename: Union[str, Path]):
        data = dict()
        for (topic_from, topic_to), transformation in self._data.items():
            if topic_from not in data:
                data[topic_from] = dict()
            data[topic_from][topic_to] = _serialize_transform(transformation)

        with open(str(filename), 'w') as file:
            yaml.dump(data, file)

    def plot_graph(self):
        plt.figure()
        nx.draw(self._graph, with_labels=True, font_weight='bold')

    def plot_frames(self, origin: str, length: float):
        if origin not in self._graph.nodes:
            raise RuntimeError(f"Frame '{origin}' does not exist.")

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for frame in self._graph.nodes:
            m = self.get(origin, frame)
            if m is None:
                continue

            points = np.array([
                [0, 0, 0],
                [length, 0, 0],
                [0, length, 0],
                [0, 0, length],
            ])
            points = m.transformCloud(points.T).T

            ax.plot([points[0, 0], points[1, 0]], [points[0, 1], points[1, 1]], [points[0, 2], points[1, 2]],
                    c='tab:red')
            ax.plot([points[0, 0], points[2, 0]], [points[0, 1], points[2, 1]], [points[0, 2], points[2, 2]],
                    c='tab:green')
            ax.plot([points[0, 0], points[3, 0]], [points[0, 1], points[3, 1]], [points[0, 2], points[3, 2]],
                    c='tab:blue')
            ax.text(points[0, 0], points[0, 1], points[0, 2], frame, size='x-small', color='black')

        # axis equal
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()
        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)
        plot_radius = 0.5 * max([x_range, y_range, z_range])
        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
