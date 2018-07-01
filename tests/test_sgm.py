import unittest
import numpy as np
from causalgraphicalmodels.examples import chain_csm


class TestSGM(unittest.TestCase):
    def test_cgm(self):
        chain_cgm = chain_csm.cgm
        self.assertTrue(chain_cgm.is_d_separated("a", "c", {"b"}))
        self.assertFalse(chain_cgm.is_d_separated("a", "c", set()))

    def test_sample(self):
        sample = chain_csm.sample(n_samples=1)

        self.assertEqual(sample.c.squeeze(), 6)

    def test_do(self):
        sample = (
            chain_csm
            .do("a")
            .sample(n_samples=1, set_values={"a": np.array([5])})
        )

        self.assertEqual(sample.c.squeeze(), 14)
