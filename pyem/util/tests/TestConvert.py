# Copyright (C) 2018 Daniel Asarnow
# University of California, San Francisco
import numpy as np
import unittest
from pyem import util

e1 = np.array([0, 0, 0])
aa1 = np.array([0, 0, 0])
r1 = np.identity(3)
q1 = np.array([1, 0, 0, 0])
# e2 = np.deg2rad(np.array([30, 60, 120]))
# aa2 = np.array([-0.97569453, -0.97569453, -2.30851804])

e3 = np.array([0.7844259044446438, 2.2296280676903204, -1.6883258657322704])
aa3 = np.array([-2.14932622394, 0.746829503588, -0.487367610323])
r3 = np.array([[0.75235481, -0.65220062, 0.09271774],
               [-0.34748354, -0.512477, -0.78525315],
               [0.55965831, 0.5585711, -0.61219352]])
q3 = np.array([0.39613265, -0.84808981, 0.294687, -0.19230748])


class TestConvert(unittest.TestCase):
    def test_expmap(self):
        r1test = util.expmap(aa1)
        self.assertTrue(np.array_equal(r1, r1test))
        r3test = util.expmap(aa3)
        self.assertTrue(np.allclose(r3, r3test))

    def test_euler2rot(self):
        r1test = util.euler2rot(*e1)
        self.assertTrue(np.array_equal(r1, r1test))
        r2test = util.euler2rot(*e3)
        self.assertTrue(np.allclose(r3, r2test))

    def test_rot2euler(self):
        e1test = util.rot2euler(r1)
        self.assertTrue(np.array_equal(e1, e1test))
        e2test = util.rot2euler(r3)
        self.assertTrue(np.allclose(e3, e2test))

    def test_euler2quat(self):
        q1test = util.euler2quat(*e1)
        self.assertTrue(np.array_equal(q1, q1test))
        q2test = util.euler2quat(*e3)
        self.assertTrue(np.allclose(q3, q2test))

    def test_quat2euler(self):
        e1test = util.quat2euler(q1)
        self.assertTrue(np.array_equal(e1, e1test))
        e2test = util.quat2euler(q3)
        self.assertTrue(np.allclose(e3, e2test))

    def test_quat2rot(self):
        r1test = util.quat2rot(q1)
        self.assertTrue(np.array_equal(r1, r1test))
        r3test = util.quat2rot(q3)
        self.assertTrue(np.allclose(r3, r3test))

    def test_rot2quat(self):
        q1test = util.rot2quat(r1)
        self.assertTrue(np.array_equal(q1, q1test))
        q3test = util.rot2quat(r3)
        self.assertTrue(np.allclose(q3, np.sign(q3) * np.abs(q3test)))

    def test_quat2aa(self):
        aa1test = util.quat2aa(q1)
        self.assertTrue(np.array_equal(aa1, aa1test))
        aa3test = util.quat2aa(q3)
        self.assertTrue(np.allclose(aa3, aa3test))

    def test_aa2quat(self):
        q1test = util.aa2quat(aa1)
        self.assertTrue(np.array_equal(q1, q1test))
        q3test = util.aa2quat(aa3)
        self.assertTrue(np.allclose(q3, q3test))


if __name__ == '__main__':
    unittest.main()
