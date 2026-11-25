import unittest
from unittest.mock import patch

import numpy as np
import torch

from diagnostics.paired_feedback_control import ACTIONS, DISCOUNT, Panel, RolloutScorer, paired_labels, proposal
from diagnostics.run_paired_feedback_control import choose
from diagnostics.validated_directions import panels, trust_weights


class PairedFeedbackTests(unittest.TestCase):
    def test_branch_projection_preserves_fork_start(self):
        class World:
            cosine_head=True
        w=torch.tensor([[1.,0.],[0.,1.],[0.,0.]])
        init=w.clone()
        delta=torch.tensor([[0.,1.],[1.,0.],[1.,1.]])
        result=proposal(World(),w,(delta,torch.zeros_like(delta)),0)
        torch.testing.assert_close(w,init)
        torch.testing.assert_close(result[-1],torch.zeros(2))
        torch.testing.assert_close(result[:-1].norm(dim=0),torch.ones(2))

    def test_private_reports_use_only_sampled_validation_clients(self):
        class World:
            evals=[[0,2]]
            val_groups=torch.tensor([0,1,2,2])
            vx=torch.tensor([[1.,0.],[100.,100.],[0.,1.],[0.,1.]])
            vy=torch.tensor([0,0,0,1])
        p=Panel.from_world(World(),0)
        report=p.reports(torch.eye(2))
        self.assertEqual(p.sizes.tolist(),[1,2])
        self.assertAlmostEqual(p.summarize(report)[0],1/3,places=6)

    def test_measured_continuation_is_discounted_and_stops_at_terminal(self):
        class PanelStub:
            def reports(self,value):
                return value
            def cost(self,value):
                return float(value)
        def step(w,model,t,a,first):
            return model-a if t==6 else model*2
        with patch('diagnostics.paired_feedback_control.advance',side_effect=step):
            labs=paired_labels(None,0,6,None,0,[PanelStub()]*8,0)
            np.testing.assert_allclose(labs[2],[2,2+DISCOUNT*2])
            terminal=paired_labels(None,0,7,None,0,[PanelStub()]*8,0)
            np.testing.assert_equal(terminal[:,0],terminal[:,1])

    def test_paired_regression_is_invariant_to_common_return_offset(self):
        rng=np.random.default_rng(241)
        xs=rng.normal(size=(8,len(ACTIONS),3))
        y=xs[:,:,1]-.1*xs[:,:,0]
        a=RolloutScorer.fit(xs,y,2)
        b=RolloutScorer.fit(xs,y+np.arange(8)[:,None],2)
        np.testing.assert_allclose(a.scores(xs[0]),b.scores(xs[0]))
        self.assertEqual(choose(np.zeros(len(ACTIONS)),4),4)


class ValidatedDirectionTests(unittest.TestCase):
    def test_equal_gains_preserve_prior_and_zero_weight_stays_zero(self):
        prior=torch.tensor([.75, .25, 0.])
        torch.testing.assert_close(trust_weights(prior,torch.ones(3)),prior)
        tilted=trust_weights(prior,torch.tensor([-1.,1.,5.]))
        self.assertLess(tilted[0],prior[0])
        self.assertEqual(float(tilted[2]),0.)
        self.assertAlmostEqual(float(tilted.sum()),1.)

    def test_proposal_and_checking_panels_are_disjoint(self):
        class World:
            evals=[list(reversed(range(16)))]
            val_groups=torch.arange(24)
            vx=torch.arange(24)[:,None]
            vy=torch.arange(24)
        a,b=panels(World(),0,'validation')
        self.assertEqual(len(a.sizes),8)
        self.assertEqual(len(b.sizes),8)
        self.assertFalse(set(a.y.tolist()) & set(b.y.tolist()))
        self.assertEqual(set(a.y.tolist()+b.y.tolist()),set(range(16)))


class NativePolicyEvaluationTests(unittest.TestCase):
    def test_identical_native_policies_share_prefix_and_final_model(self):
        from diagnostics import fit_native_step_control as native
        class Stub:
            global_model=None
            schedule=[[0]]*40
            manifest={k:'same' for k in ('initial_model_sha256','native_data_sha256','private_splits_sha256')}
            def record_test(self,model):
                return {'accuracy':model,'worst10':model}
        def advance(w,model,t,a,schedule,first):
            return model+1+a
        with patch.object(native,'open_world',return_value=(Stub(),['none']*40)), \
             patch.object(native,'vector',return_value=0), \
             patch.object(native,'probe',return_value=([],None,None)), \
             patch.object(native,'commit',side_effect=advance) as commit:
            result=native.shared_evaluation(({},10,None,{'a':{'arm':1},'b':{'arm':1}},True))
        self.assertEqual(commit.call_count,8)
        self.assertEqual(result['a']['metrics'],result['b']['metrics'])
        self.assertEqual(result['a']['actions'],[1]*8)
