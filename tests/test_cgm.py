import unittest
from causalgraphicalmodels.examples import sprinkler, simple_confounded, \
    simple_confounded_hidden_confounder, front_door_example


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

    def test_get_all_backdoor_adjustment_sets_hidden(self):
        self.assertFalse(
            simple_confounded_hidden_confounder
                .get_all_backdoor_adjustment_sets("x", "y"))

    def test_is_valid_adjustment_set(self):
        self.assertTrue(
            simple_confounded.is_valid_backdoor_adjustment_set("x", "y", {"z"}))
        self.assertFalse(
            simple_confounded.is_valid_backdoor_adjustment_set("x", "y", set()))
        self.assertFalse(
            simple_confounded_hidden_confounder
                .is_valid_backdoor_adjustment_set("x", "y", set()))

    def test_get_all_independence_relationships(self):
        self.assertFalse(
            simple_confounded_hidden_confounder
                .get_all_independence_relationships())

    def test_is_valid_frontdoor_adjustment_set(self):
        self.assertTrue(
            front_door_example.is_valid_frontdoor_adjustment_set("x", "y", "z"))

        self.assertFalse(
            simple_confounded.is_valid_frontdoor_adjustment_set("x", "y", "z"))

