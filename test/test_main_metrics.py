#!/usr/bin/env python
"""
unit test for the pure functional metric APIs in main_ape and main_rpe
author: Michael Grupp

This file is part of evo (github.com/MichaelGrupp/evo).

evo is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

evo is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with evo.  If not, see <http://www.gnu.org/licenses/>.
"""

import copy
import unittest

import numpy as np

import helpers
from evo.core import metrics
from evo.core.trajectory import Plane
from evo.main_ape import ape
from evo.main_rpe import rpe


class TestApeApi(unittest.TestCase):
    def setUp(self):
        self.traj_ref = helpers.fake_trajectory(100, 0.1)
        self.traj_est = helpers.fake_trajectory(100, 0.1)

    def test_pure_no_side_effects(self):
        traj_ref_copy = copy.deepcopy(self.traj_ref)
        traj_est_copy = copy.deepcopy(self.traj_est)
        ape(
            self.traj_ref,
            self.traj_est,
            metrics.PoseRelation.translation_part,
            align=True,
            correct_scale=True,
            project_to_plane=Plane.xy,
        )
        self.assertEqual(self.traj_ref, traj_ref_copy)
        self.assertEqual(self.traj_est, traj_est_copy)

    def test_bit_identical_to_metrics_class(self):
        result = ape(
            self.traj_ref, self.traj_est, metrics.PoseRelation.translation_part
        )
        ape_metric = metrics.APE(metrics.PoseRelation.translation_part)
        ape_metric.process_data((self.traj_ref, self.traj_est))
        self.assertEqual(result.stats, ape_metric.get_all_statistics())
        self.assertTrue(
            np.array_equal(result.np_arrays["error_array"], ape_metric.error)
        )

    def test_bit_identical_with_alignment(self):
        result = ape(
            self.traj_ref,
            self.traj_est,
            metrics.PoseRelation.full_transformation,
            align=True,
            correct_scale=True,
        )
        # Replicate the aligned computation manually on copies.
        traj_est_aligned = copy.deepcopy(self.traj_est)
        traj_est_aligned.align(self.traj_ref, correct_scale=True)
        ape_metric = metrics.APE(metrics.PoseRelation.full_transformation)
        ape_metric.process_data((self.traj_ref, traj_est_aligned))
        self.assertEqual(result.stats, ape_metric.get_all_statistics())
        self.assertTrue(
            np.array_equal(result.np_arrays["error_array"], ape_metric.error)
        )

    def test_deterministic_repeated_calls(self):
        kwargs = {"align": True, "correct_scale": True}
        result_1 = ape(
            self.traj_ref,
            self.traj_est,
            metrics.PoseRelation.translation_part,
            **kwargs,
        )
        result_2 = ape(
            self.traj_ref,
            self.traj_est,
            metrics.PoseRelation.translation_part,
            **kwargs,
        )
        self.assertEqual(result_1, result_2)


class TestRpeApi(unittest.TestCase):
    def setUp(self):
        self.traj_ref = helpers.fake_trajectory(100, 0.1)
        self.traj_est = helpers.fake_trajectory(100, 0.1)

    def test_pure_no_side_effects(self):
        traj_ref_copy = copy.deepcopy(self.traj_ref)
        traj_est_copy = copy.deepcopy(self.traj_est)
        rpe(
            self.traj_ref,
            self.traj_est,
            metrics.PoseRelation.translation_part,
            delta=1,
            delta_unit=metrics.Unit.frames,
            align=True,
            correct_scale=True,
            project_to_plane=Plane.xy,
        )
        self.assertEqual(self.traj_ref, traj_ref_copy)
        self.assertEqual(self.traj_est, traj_est_copy)

    def test_bit_identical_to_metrics_class(self):
        result = rpe(
            self.traj_ref,
            self.traj_est,
            metrics.PoseRelation.translation_part,
            delta=1,
            delta_unit=metrics.Unit.frames,
        )
        rpe_metric = metrics.RPE(
            metrics.PoseRelation.translation_part,
            delta=1,
            delta_unit=metrics.Unit.frames,
        )
        rpe_metric.process_data((self.traj_ref, self.traj_est))
        self.assertEqual(result.stats, rpe_metric.get_all_statistics())
        self.assertTrue(
            np.array_equal(result.np_arrays["error_array"], rpe_metric.error)
        )

    def test_bit_identical_with_alignment(self):
        result = rpe(
            self.traj_ref,
            self.traj_est,
            metrics.PoseRelation.rotation_angle_deg,
            delta=1,
            delta_unit=metrics.Unit.frames,
            align=True,
            correct_scale=True,
        )
        # Replicate the aligned computation manually on copies.
        traj_est_aligned = copy.deepcopy(self.traj_est)
        traj_est_aligned.align(self.traj_ref, correct_scale=True)
        rpe_metric = metrics.RPE(
            metrics.PoseRelation.rotation_angle_deg,
            delta=1,
            delta_unit=metrics.Unit.frames,
        )
        rpe_metric.process_data((self.traj_ref, traj_est_aligned))
        self.assertEqual(result.stats, rpe_metric.get_all_statistics())
        self.assertTrue(
            np.array_equal(result.np_arrays["error_array"], rpe_metric.error)
        )

    def test_deterministic_repeated_calls(self):
        kwargs = {"align": True, "correct_scale": True}
        result_1 = rpe(
            self.traj_ref,
            self.traj_est,
            metrics.PoseRelation.translation_part,
            delta=1,
            delta_unit=metrics.Unit.frames,
            **kwargs,
        )
        result_2 = rpe(
            self.traj_ref,
            self.traj_est,
            metrics.PoseRelation.translation_part,
            delta=1,
            delta_unit=metrics.Unit.frames,
            **kwargs,
        )
        self.assertEqual(result_1, result_2)


if __name__ == "__main__":
    unittest.main()
