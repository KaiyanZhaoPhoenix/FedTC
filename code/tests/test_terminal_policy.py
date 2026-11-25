import unittest

from diagnostics.audit_terminal_policy_mnist import keep_candidate
from diagnostics.confirm_terminal_timing_mnist import duration, train_timing
from diagnostics.run_terminal_policy_mnist import decide, train


class TerminalPolicyTests(unittest.TestCase):
    def test_first_decision_uses_deployed_last_policy(self):
        features = [[0.], [0.], [0.]]
        costs = [[.7, .9, .8], [.8, .15, .9], [.6, .9, .9]]
        tree = {"features": features, "leaves": [
            {"features": features, "costs": row} for row in costs
        ]}
        policy = train([tree], margin=0.)
        # Averaged across leaves, tau index 1 wins the last stage. Its real
        # continuation yields costs [.9,.15,.9] for the preceding stage.
        self.assertEqual(decide(tree, policy), (1, 1))
        self.assertEqual(decide(tree, train([tree], margin=1.)), (0, 0))

    def test_stopping_policy_accounts_for_cost_of_early_and_late_forks(self):
        features = [[0.], [0.], [0.]]
        tree = {"features": features, "leaves": [
            {"features": features, "costs": costs}
            for costs in ([.5, .5, .4], [.8, .8, .8], [.8, .8, .395])
        ]}
        # A ten-round fork gives the larger raw benefit. At this price the
        # five-round fork leaves a higher net gain; a still higher price selects neither.
        self.assertEqual(duration(tree, train_timing([tree], 0.)), 10)
        self.assertEqual(duration(tree, train_timing([tree], .01)), 5)
        self.assertEqual(duration(tree, train_timing([tree], .03)), 0)

    def test_terminal_tie_keeps_the_reference(self):
        tree = {"leaves": [{"costs":[.1, .2, .3]}, {"costs":[.2, .1, .05]}]}
        self.assertEqual(keep_candidate(tree, (1, 1)), (0, 0))
        self.assertEqual(keep_candidate(tree, (1, 2)), (1, 2))


if __name__ == '__main__':
    unittest.main()
