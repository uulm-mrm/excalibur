import motion3d as m3d
import numpy as np


def gen_Mlist(transforms_a, transforms_b, version=1, normalize=True):
    # prepare poses
    assert transforms_a.size() == transforms_b.size()
    transforms_a = transforms_a.asType(m3d.TransformType.kDualQuaternion)
    transforms_b = transforms_b.asType(m3d.TransformType.kDualQuaternion)

    if normalize:
        transforms_a.normalized_()
        transforms_b.normalized_()

    # iterate poses
    Mlist = list()
    for Pa, Pb in zip(transforms_a, transforms_b):
        # combine transforms
        Pc = Pa / Pb

        if version == 0:
            c = Pc.normalized_().getDualQuaternion().toArray()
            M = np.column_stack((np.eye(8), c))
        elif version == 1:
            C = Pc.inverse().normalized_().getDualQuaternion().toPositiveMatrix()
            identity = np.zeros(8)
            identity[0] = 1.0
            M = np.column_stack((C, identity))
        else:
            raise NotImplementedError(f"Version {version} is not implemented")

        # store result
        Mlist.append(M)

    return Mlist
