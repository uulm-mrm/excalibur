import motion3d as m3d


def container_subset(container, indices):
    new_container = m3d.TransformContainer(has_poses=container.hasPoses(), has_stamps=container.hasStamps())
    for idx in indices:
        if container.hasStamps():
            new_container.append(container.stamp_at(idx), container[idx])
        else:
            new_container.append(container[idx])
    return new_container


def iterate_uniform(container):
    if container.hasStamps():
        for stamp, transform in container.items():
            yield stamp, transform
    else:
        for transform in container:
            yield None, transform
