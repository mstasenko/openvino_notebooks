"""Dependency-free contract test for the portrait-only CPU validation case."""

import unittest

from portrait_chunked_cpu_validation import PORTRAIT_SHAPE, select_case_shapes


class PortraitCaseSelectionTest(unittest.TestCase):
    def test_selects_only_actual_t2_portrait_shape(self) -> None:
        shapes = select_case_shapes()
        self.assertEqual(shapes, ((2, 240, 135),))
        self.assertEqual(shapes[0], PORTRAIT_SHAPE)
        self.assertEqual(shapes[0][0], 2)
        self.assertGreater(shapes[0][1], shapes[0][2])
        self.assertNotIn((2, 135, 240), shapes)


if __name__ == "__main__":
    unittest.main()
