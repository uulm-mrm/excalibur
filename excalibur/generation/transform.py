from enum import auto, Enum
import numpy as np

import motion3d as m3d


def random_directions(n: int):
    directions = np.random.normal(size=(n, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return directions


class RandomType(Enum):
    CONST = auto()
    UNIFORM = auto()
    NORMAL = auto()

    def eval(self, n, value):
        if self == RandomType.CONST:
            return np.ones(n) * value
        elif self == RandomType.UNIFORM:
            return np.random.uniform(low=-value, high=value, size=n)
        elif self == RandomType.NORMAL:
            return np.random.normal(loc=0, scale=value, size=n)
        else:
            raise NotImplementedError("Random type not implemented")


def random_transforms(n: int, trans_value: float, trans_type: RandomType, rot_value: float, rot_type: RandomType):
    # translations
    trans_dir = random_directions(n)
    trans_norm = trans_type.eval(n, trans_value)
    translation = np.einsum('nk,n->nk', trans_dir, trans_norm)

    # rotations
    rot_axis = random_directions(n)
    rot_angle = rot_type.eval(n, rot_value)

    # transformations
    trafos = [m3d.AxisAngleTransform(translation[i, :], rot_angle[i], rot_axis[i, :]).normalized_()
              for i in range(n)]
    container = m3d.TransformContainer(trafos, has_poses=True)
    return container


def random_transform(trans_value: float, trans_type: RandomType, rot_value: float, rot_type: RandomType):
    return random_transforms(1, trans_value, trans_type, rot_value, rot_type)[0]
