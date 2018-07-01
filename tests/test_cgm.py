import unittest
from causalgraphicalmodels.examples import sprinkler, simple_confounded


class TestCGM(unittest.TestCase):
    def test_is_d_separated(self):
        s = sprinkler

        self.assertTrue(s.is_d_separated("season", "slippery", {"wet"}))
        self.assertTrue(s.is_d_separated("season", "slippery",
                                         {"rain", "sprinkler"}))
        self.assertTrue(s.is_d_separated("season", "slippery",
                                         {"rain", "season", "wet"}))

        self.assertFalse(s.is_d_separated("rain", "sprinkler"))
        self.assertTrue(s.is_d_separated("rain", "sprinkler", {"season"}))
        self.assertFalse(s.is_d_separated("rain", "sprinkler", {"wet"}))

    def test_get_all_backdoor_adjustment_sets(self):
        s = simple_confounded.get_all_backdoor_adjustment_sets("x", "y")
        expected_results = frozenset([frozenset(["z"])])
        self.assertEqual(s, expected_results)

    def test_is_valid_adjustment_set(self):
        self.assertTrue(
            simple_confounded.is_valid_adjustment_set("x", "y", {"z"}))
        self.assertFalse(
            simple_confounded.is_valid_adjustment_set("x", "y", set()))
