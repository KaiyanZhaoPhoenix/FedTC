import unittest
import numpy as np

from diagnostics.run_stage_a import Critic, gain
from diagnostics.stage_a_world import GRID, WINDOWS, dirichlet


class StageATests(unittest.TestCase):
    def test_future_reward_changes_only_discounted_choice(self):
        def state(signal,step):
            x=np.zeros(10)
            x[0],x[7]=signal,step/WINDOWS
            return x.tolist()

        def row(x,a,reward,y,t):
            return dict(s=x,a=a,low=reward-2,high=reward+2,slack=2,next=y,t=t)

        initial=state(0,6)
        data=[]
        for a in range(len(GRID)):
            data.append(row(initial,a,.1 if a==0 else 0.,state(1 if a==1 else -1,7),6))
            for v in (-1,1):
                data.append(row(state(v,7),a,float(v==1),state(0,8),7))
        norm=np.zeros(10),np.ones(10)
        self.assertAlmostEqual(gain(data[0]),.1)
        self.assertEqual(Critic(data*64,0,norm).scores(initial).argmax(),0)
        self.assertEqual(Critic(data*64,.9,norm).scores(initial).argmax(),1)

    def test_dirichlet_partition_preserves_every_example_once(self):
        y=np.repeat(np.arange(100),20)
        groups=dirichlet(y,41)
        np.testing.assert_array_equal(np.sort(np.concatenate(groups)),np.arange(len(y)))
        self.assertEqual(len(groups),24)
        self.assertGreaterEqual(min(map(len,groups)),32)
