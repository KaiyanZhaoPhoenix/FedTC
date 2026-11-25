import unittest

import numpy as np

from diagnostics.compare_terminal_mnist import ablate, decision_audit
from diagnostics.fit_verified_continuation_mnist import continuation_costs, terminal_choice
from diagnostics.refit_last_mnist import fit_last
from diagnostics.run_terminal_policy_mnist import advantages, train


class TerminalAblationTests(unittest.TestCase):
    def test_first_action_target_compares_terminal_branches_from_its_own_successor(self):
        leaves = [{"costs": costs} for costs in ([.8, .01, .7], [.4, .02, .6], [.5, .03, .5])]
        self.assertEqual(continuation_costs({"leaves": leaves}), [.7, .4, .5])
        self.assertEqual(terminal_choice(leaves[-1]), 0)

    def test_shared_difference_prediction_is_invariant_to_a_common_offset(self):
        features = np.array([[1., 7.], [2., 7.], [3., 7.]])
        trees = [{"leaves": [{"features": features, "costs": [.4, .3, .2]}]}]
        head = fit_last(trees, "shared_difference_ridge")
        score = advantages(head, features)
        self.assertEqual(score[0], 0.)
        self.assertGreater(score[2], score[1])
        np.testing.assert_allclose(score, advantages(head, features + [18., -3.]))

    def test_private_cost_controls_actions_and_branch_selection(self):
        features = [[0.], [0.], [0.]]
        leaves = [{"features": features, "costs": costs}
                  for costs in ([.7, .9, .8], [.8, .15, .9], [.6, .9, .9])]
        tree = {"seed": 1, "features": features, "leaves": leaves}
        policy = train([tree], margin=0.)
        stages = decision_audit(tree, policy)
        self.assertEqual([stage["selected_tau"] for stage in stages], [1.25, 1.25])
        self.assertAlmostEqual(stages[0]["measured_gain"][1], .9-.15)
        self.assertAlmostEqual(stages[1]["measured_gain"][1], .8-.15)
        self.assertTrue(all(stage["selected_is_best"] for stage in stages))
        for leaf in leaves:
            leaf["test"] = [dict(accuracy=0., worst10=0.) for _ in range(3)]
        leaves[0]["test"][0]["accuracy"] = 1.
        row = ablate(tree, policy)
        self.assertEqual(row["learned_verified"]["accuracy"], 0.)
        self.assertEqual(row["matched_verified"]["accuracy"], 1.)
        self.assertEqual(row["learned_verified"]["extra_rounds"], row["matched_verified"]["extra_rounds"])


if __name__ == "__main__":
    unittest.main()
