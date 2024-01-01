from typing import List, Optional

import motion3d as m3d
import numpy as np

from .base import BaseProcessing
from excalibur.generation.transform import random_transforms, RandomType
from excalibur.utils.motion3d import iterate_uniform


class InterpolatePoses(BaseProcessing):
    def __init__(self, max_diff_nsec: Optional[int] = None, max_factor: Optional[float] = None):
        self._max_diff_nsec = max_diff_nsec
        self._max_factor = max_factor

    @staticmethod
    def _interpolate(a, b, factor):
        assert 0.0 <= factor <= 1.0
        a = a.asType(m3d.TransformType.kQuaternion)
        b = b.asType(m3d.TransformType.kQuaternion)

        t = a.getTranslation() * factor + b.getTranslation() * (1.0 - factor)
        q = a.getQuaternion().slerp(factor, b.getQuaternion())

        return m3d.QuaternionTransform(t, q).normalized_()

    def __call__(self, data):
        assert len(data) == 2, "only container pairs are supported"
        main_poses = data[0]
        sub_poses = data[1]

        assert main_poses.hasStamps() and main_poses.hasPoses(), "all containers must have stamps and poses"
        assert sub_poses.hasStamps() and sub_poses.hasPoses(), "all containers must have stamps and poses"

        main_poses_out = m3d.TransformContainer(has_stamps=True, has_poses=True)
        sub_poses_out = m3d.TransformContainer(has_stamps=True, has_poses=True)

        for main_stamp, main_pose in main_poses.items():
            # find surrounding stamps
            prev_stamp, prev_pose = sub_poses.find_le(main_stamp)
            next_stamp, next_pose = sub_poses.find_gt(main_stamp)

            # check stamps
            if prev_stamp is None:
                continue

            prev_stamp_diff_nsec = main_stamp.toNSec() - prev_stamp.toNSec()
            if prev_stamp_diff_nsec == 0:
                # use prev_pose
                main_poses_out.insert(main_stamp, main_pose)
                sub_poses_out.insert(main_stamp, prev_pose)

            elif next_stamp is not None:
                # interpolate
                factor = prev_stamp_diff_nsec / (next_stamp.toNSec() - prev_stamp.toNSec())

                # check
                if self._max_factor is not None and \
                        self._max_factor < factor < 1.0 - self._max_factor:
                    continue

                next_stamp_diff_nsec = next_stamp.toNSec() - main_stamp.toNSec()
                if self._max_diff_nsec is not None and \
                        min(prev_stamp_diff_nsec, next_stamp_diff_nsec) > self._max_diff_nsec:
                    continue

                # calculate
                interp_pose = self._interpolate(prev_pose, next_pose, factor)
                main_poses_out.insert(main_stamp, main_pose)
                sub_poses_out.insert(main_stamp, interp_pose)

            else:
                break

        return main_poses_out, sub_poses_out


class AddStampOffsets(BaseProcessing):
    def __init__(self, offsets_sec: List[float]):
        self._offsets_nsec = [int(o * 1e9) if o is not None else None
                              for o in offsets_sec]

    def __call__(self, data):
        # check offset length
        assert len(data) == len(self._offsets_nsec), "the number of offsets must match the number of containers"

        # initialize output
        data_out = []

        # iterate containers
        for container, offset in zip(data, self._offsets_nsec):
            assert container.hasStamps(), "all input containers must have stamps"

            # passthrough if no offset is given
            if offset is None or offset == 0:
                data_out.append(container)
                continue

            # add offset to stamps
            container_out = m3d.TransformContainer(has_stamps=True, has_poses=container.hasPoses())
            for stamp, transform in container.items():
                stamp_out = m3d.Time.FromNSec(stamp.toNSec() + offset)
                container_out.append(stamp_out, transform)

            data_out.append(container_out)

        return data_out


class MatchPoses(BaseProcessing):
    def __init__(self, max_diff_nsec: Optional[int] = None, unique: bool = True):
        self._max_diff_nsec = max_diff_nsec
        self._unique = unique

    def __call__(self, data):
        assert len(data) == 2, "only container pairs are supported"
        main_poses = data[0]
        sub_poses = data[1]

        assert main_poses.hasPoses() and sub_poses.hasPoses(), "all containers must provide poses"

        # check for stamps
        if not main_poses.hasStamps():
            assert not sub_poses.hasStamps(), "all containers must have either stamps or no stamps"
            assert len(main_poses) == len(sub_poses), "container lengths must match if no stamps are provided"
            return main_poses, sub_poses

        assert sub_poses.hasStamps(), "all containers must have either stamps or no stamps"

        # find closest stamps
        stamp_pairs = {}
        for main_stamp in main_poses.stamps():
            sub_stamp, _ = sub_poses.find_closest(main_stamp)

            # check difference
            if self._max_diff_nsec is None or m3d.timeDiffAbs(main_stamp, sub_stamp).toNSec() <= self._max_diff_nsec:
                stamp_pairs[main_stamp] = sub_stamp

        if self._unique:
            # invert stamp pairs
            stamp_pairs_inv = {}
            for main_stamp, sub_stamp in stamp_pairs.items():
                if sub_stamp in stamp_pairs_inv:
                    stamp_pairs_inv[sub_stamp].append(main_stamp)
                else:
                    stamp_pairs_inv[sub_stamp] = [main_stamp]

            # find closest of inverse association
            stamp_pairs_unique = {}
            for sub_stamp, main_stamp_list in stamp_pairs_inv.items():
                # calculate diffs for all stamps
                stamp_diffs = [m3d.timeDiffAbs(main_stamp, sub_stamp).toNSec() for main_stamp in main_stamp_list]

                # select  closest
                best_main_stamp_idx = np.argmin(stamp_diffs)
                stamp_pairs_unique[main_stamp_list[best_main_stamp_idx]] = sub_stamp
            stamp_pairs = stamp_pairs_unique

        # generate output
        main_poses_out = m3d.TransformContainer(has_stamps=True, has_poses=True)
        sub_poses_out = m3d.TransformContainer(has_stamps=True, has_poses=True)
        for main_stamp, sub_stamp in stamp_pairs.items():
            main_poses_out.insert(main_stamp, main_poses.at_stamp(main_stamp))
            sub_poses_out.insert(sub_stamp, sub_poses.at_stamp(sub_stamp))

        return main_poses_out, sub_poses_out


class MergeTransforms(BaseProcessing):
    def __init__(self):
        pass

    def __call__(self, data_in):
        # check input
        if isinstance(data_in, dict):
            iter_data = data_in.values()
        elif isinstance(data_in, list):
            iter_data = data_in
        else:
            raise RuntimeError("Unsupported input type")

        if len(iter_data) == 0:
            raise RuntimeError("Input data must not be empty")

        # initialize output
        data_out = data_in[0]
        if not isinstance(data_out, (list, tuple)):
            raise RuntimeError("Input items must be iterable")

        # iterate input
        for data_item in data_in[1:]:
            if not isinstance(data_item, (list, tuple)):
                raise RuntimeError("Input items must be iterable")

            for i in range(len(data_item)):
                data_out[i].extend(data_item[i])

        return data_out


class PivotTransforms(BaseProcessing):
    def __init__(self):
        pass

    def __call__(self, data_in):
        # check input
        if isinstance(data_in, dict):
            iter_data = data_in.values()
        elif isinstance(data_in, list):
            iter_data = data_in
        else:
            raise RuntimeError("Unsupported input type")

        if len(iter_data) == 0:
            raise RuntimeError("Input data must not be empty")

        # initialize output
        if not isinstance(data_in[0], (list, tuple)):
            raise RuntimeError("Input items must be iterable")
        data_out = [[d] for d in data_in[0]]

        # iterate input
        for data_item in data_in[1:]:
            if not isinstance(data_item, (list, tuple)):
                raise RuntimeError("Input items must be iterable")

            for i in range(len(data_item)):
                data_out[i].append(data_item[i])

        return tuple(data_out)


class Normalized(BaseProcessing):
    def __init__(self, inplace: bool = True):
        self._inplace = inplace

    def __call__(self, transforms):
        if self._inplace:
            for t in transforms:
                t.normalized_()
            return transforms
        else:
            return [t.normalized() for t in transforms]


class ApplyNoise(BaseProcessing):
    def __init__(self, trans_stddevs: List[float], rot_stddevs: List[float]):
        self._trans_stddevs = trans_stddevs
        self._rot_stddevs = rot_stddevs

    def __call__(self, transforms_list):
        assert len(transforms_list) == len(self._trans_stddevs) == len(self._rot_stddevs), \
            "the numbers of containers, translation, and rotation standard deviations must match"

        # iterate transforms
        transforms_list_out = []
        for transforms, trans_stddev, rot_stddev in zip(transforms_list, self._trans_stddevs, self._rot_stddevs):
            # generate noise
            noise = random_transforms(len(transforms), trans_stddev, RandomType.NORMAL, rot_stddev, RandomType.NORMAL)

            # apply noise
            transforms_noise = transforms.copy()
            for idx in range(len(transforms)):
                transforms_noise[idx] *= noise[idx]

            # store transforms
            transforms_list_out.append(transforms_noise)

        return transforms_list_out


class PosesToMotions(BaseProcessing):
    def __init__(self, min_step_sec: Optional[float] = None, max_step_sec: Optional[float] = None):
        self._min_step_sec = min_step_sec
        self._max_step_sec = max_step_sec

    def _proc_without_stamps(self, poses1, poses2):
        motions1_out = poses1.removeStamps().asMotions_()
        motions2_out = poses2.removeStamps().asMotions_()
        return motions1_out, motions2_out

    def _proc_with_min_step(self, poses1, poses2):
        assert poses1.hasStamps() and poses2.hasStamps()

        # initialize output
        motions1_out = m3d.TransformContainer(has_stamps=False, has_poses=False)
        motions2_out = m3d.TransformContainer(has_stamps=False, has_poses=False)

        # iterate poses
        for first_idx in range(len(poses1)):
            stamp_first = poses1.stamp_at(first_idx).toSec()

            for last_idx in range(first_idx + 1, len(poses1)):
                stamp_last = poses1.stamp_at(last_idx).toSec()
                stamp_diff = stamp_last - stamp_first
                if stamp_diff < self._min_step_sec:
                    continue
                if stamp_diff > self._max_step_sec:
                    break
                motions1_out.append(poses1.at(first_idx).inverse() * poses1.at(last_idx))
                motions2_out.append(poses2.at(first_idx).inverse() * poses2.at(last_idx))
                break

        # return result
        return motions1_out, motions2_out

    def _proc_without_min_step(self, poses1, poses2):
        assert poses1.hasStamps() and poses2.hasStamps()

        # stamps
        stamps = np.array([t.toSec() for t in poses1.stamps()])
        steps = stamps[1:] - stamps[:-1]

        # motions
        motions1 = poses1.removeStamps().asMotions_()
        motions2 = poses2.removeStamps().asMotions_()

        # remove large time steps
        motions1_out = m3d.TransformContainer(has_stamps=False, has_poses=False)
        motions2_out = m3d.TransformContainer(has_stamps=False, has_poses=False)

        for step, m1, m2 in zip(steps, motions1, motions2):
            assert step > 0.0
            if step <= self._max_step_sec:
                motions1_out.append(m1)
                motions2_out.append(m2)

        return motions1_out, motions2_out

    def __call__(self, data):
        assert len(data) == 2, "only container pairs are supported"
        poses1 = data[0]
        poses2 = data[1]

        assert poses1.hasPoses() and poses2.hasPoses(), "all containers must provide poses"

        if not poses1.hasStamps() or not poses2.hasStamps():
            return self._proc_without_stamps(poses1, poses2)

        if self._min_step_sec is None:
            if self._max_step_sec is None:
                return self._proc_without_stamps(poses1, poses2)
            return self._proc_without_min_step(poses1, poses2)

        return self._proc_with_min_step(poses1, poses2)


class PosesToPoints(BaseProcessing):
    def __init__(self):
        pass

    def __call__(self, data):
        assert len(data) == 2, "only container pairs are supported"
        poses1 = data[0]
        poses2 = data[1]

        assert poses1.hasPoses() and poses2.hasPoses(), "all containers must provide poses"

        points1 = np.array([p.getTranslation() for p in poses1])
        points2 = np.array([p.getTranslation() for p in poses2])
        return points1, points2


class TransformFilter(BaseProcessing):
    def __init__(self, min_translation: Optional[float] = None, max_translation: Optional[float] = None,
                 min_rotation_deg: Optional[float] = None, max_rotation_deg: Optional[float] = None):
        self._min_translation = min_translation
        self._max_translation = max_translation
        self._min_rotation_rad = np.deg2rad(min_rotation_deg) if min_rotation_deg is not None else None
        self._max_rotation_rad = np.deg2rad(max_rotation_deg) if max_rotation_deg is not None else None

    def __call__(self, data):
        assert len(data) > 0, "at least one container is required"

        # assert that all data containers have equal settings
        data_len = len(data[0])
        has_stamps = data[0].hasStamps()
        has_poses = data[0].hasPoses()
        for d in data[1:]:
            assert len(d) == data_len and d.hasStamps() == has_stamps and d.hasPoses() == has_poses

        # initialize output
        data_out = [m3d.TransformContainer(has_stamps=has_stamps, has_poses=has_poses) for _ in range(len(data))]

        # iterate transforms
        for transform_list in zip(*[iterate_uniform(container) for container in data]):

            # check transforms
            skip = False
            for stamp, t in transform_list:
                translation_norm = t.translationNorm()
                rotation_norm = t.rotationNorm()

                # check translation
                if self._min_translation is not None and translation_norm < self._min_translation:
                    skip = True
                    break

                if self._max_translation is not None and translation_norm > self._max_translation:
                    skip = True
                    break

                # check rotation
                if self._min_rotation_rad is not None and rotation_norm < self._min_rotation_rad:
                    skip = True
                    break

                if self._max_rotation_rad is not None and rotation_norm > self._max_rotation_rad:
                    skip = True
                    break

            if skip:
                continue

            # append if success
            for i, (stamp, t) in enumerate(transform_list):
                if stamp is None:
                    data_out[i].append(t)
                else:
                    data_out[i].insert(stamp, t)

        return data_out


class RemoveMotionOutliers(BaseProcessing):
    def __init__(self, max_rotation_diff_deg: Optional[float] = None):
        self._max_rotation_diff_rad = np.deg2rad(max_rotation_diff_deg) if max_rotation_diff_deg is not None else None

    def __call__(self, data):
        assert len(data) == 2, "only container pairs are supported"
        motions1 = data[0]
        motions2 = data[1]

        assert motions1.hasMotions() and motions2.hasMotions(), "all containers must provide motions"
        assert not motions1.hasStamps() and not motions2.hasStamps(), "all containers must provide not stamps"

        motions1_out = m3d.TransformContainer(has_stamps=False, has_poses=False)
        motions2_out = m3d.TransformContainer(has_stamps=False, has_poses=False)

        for m1, m2 in zip(motions1, motions2):
            # check rotation offset
            rotation_diff_deg = np.abs(m1.rotationNorm() - m2.rotationNorm())
            if self._max_rotation_diff_rad is not None and \
                    rotation_diff_deg > self._max_rotation_diff_rad:
                continue

            motions1_out.append(m1)
            motions2_out.append(m2)

        return motions1_out, motions2_out
