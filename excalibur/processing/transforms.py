import motion3d as m3d
import numpy as np

from .base import BaseProcessing


class InterpolatePoses(BaseProcessing):
    def __init__(self, max_diff_nsec=None, max_factor=None):
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
        assert len(data) == 2
        main_poses = data[0]
        sub_poses = data[1]

        assert main_poses.hasStamps() and main_poses.hasPoses()
        assert sub_poses.hasStamps() and sub_poses.hasPoses()

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


class MatchPoses(BaseProcessing):
    def __init__(self, max_diff_nsec=1000):
        self._max_diff_nsec = max_diff_nsec

    def __call__(self, data):
        assert len(data) == 2
        main_poses = data[0]
        sub_poses = data[1]

        assert main_poses.hasStamps() and main_poses.hasPoses()
        assert sub_poses.hasStamps() and sub_poses.hasPoses()

        main_poses_out = m3d.TransformContainer(has_stamps=True, has_poses=True)
        sub_poses_out = m3d.TransformContainer(has_stamps=True, has_poses=True)

        for main_stamp, main_pose in main_poses.items():
            # find closest stamps
            match_stamp, match_pose = sub_poses.find_closest(main_stamp)

            # check stamps
            if m3d.timeDiffAbs(main_stamp, match_stamp).toNSec() <= self._max_diff_nsec:
                main_poses_out.insert(main_stamp, main_pose)
                sub_poses_out.insert(main_stamp, match_pose)

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


class Normalized(BaseProcessing):
    def __init__(self, inplace=True):
        self._inplace = inplace

    def __call__(self, transforms):
        if self._inplace:
            for t in transforms:
                t.normalized_()
            return transforms
        else:
            return [t.normalized() for t in transforms]


class PosesToMotions(BaseProcessing):
    def __init__(self, max_step_sec=None):
        self._max_step_sec = max_step_sec

    def __call__(self, data):
        assert len(data) == 2
        poses1 = data[0]
        poses2 = data[1]

        assert poses1.hasPoses() and poses2.hasPoses()

        if self._max_step_sec is None:
            motions1_out = poses1.removeStamps().asMotions_()
            motions2_out = poses2.removeStamps().asMotions_()
            return motions1_out, motions2_out

        else:
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


class RemoveMotionOutliers(BaseProcessing):
    def __init__(self, max_translation, max_rotation_diff_deg):
        self._max_translation = max_translation
        self._max_rotation_diff_rad = np.deg2rad(max_rotation_diff_deg)

    def __call__(self, data):
        assert len(data) == 2
        motions1 = data[0]
        motions2 = data[1]

        assert motions1.hasMotions()
        assert motions2.hasMotions()

        motions1_out = m3d.TransformContainer(has_stamps=False, has_poses=False)
        motions2_out = m3d.TransformContainer(has_stamps=False, has_poses=False)

        for m1, m2 in zip(motions1, motions2):
            # translation
            if m1.translationNorm() > self._max_translation or m2.translationNorm() > self._max_translation:
                continue

            # rotation
            rotation_diff_deg = np.abs(m1.rotationNorm() - m2.rotationNorm())
            if rotation_diff_deg > self._max_rotation_diff_rad:
                continue

            motions1_out.insert(m1)
            motions2_out.insert(m2)

        return motions1_out, motions2_out
