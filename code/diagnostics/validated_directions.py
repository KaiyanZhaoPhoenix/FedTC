"""Use independent client validation evidence to reweight a common upload cohort."""
import torch
import torch.nn.functional as F
from diagnostics.paired_feedback_control import Panel


def panels(world, macro, family):
    if family=='aggregate':
        return None,Panel.from_world(world,macro)
    ids=world.evals[macro]
    half=len(ids)//2
    return Panel.from_world(world,macro,ids[:half]),Panel.from_world(world,macro,ids[half:])


def trust_weights(prior, gains):
    spread=gains.std(unbiased=False)
    z=((gains-gains.mean())/spread.clamp_min(1e-7)).clamp(-2,2)
    return (prior.log()+.8*z).softmax(0)


def options(world,start,i,scale,proposal_panel,selection_panel):
    from diagnostics.round_feedback_mnist import round_options
    if proposal_panel is None:
        return round_options(world,start,i,scale,selection_panel)
    world.w=start
    updates,_,_=world.uploads(i,i>=20)
    sizes=updates.new_tensor([len(world.train[c]) for c in world.cohorts[i]])
    prior=sizes/sizes.sum()
    # Each evaluator computes 12 directional derivatives locally. The server
    # receives these scalars. The training utility and participant identities
    # used by the contamination simulator do not enter the weighting rule.
    logits=proposal_panel.x@start
    residual=F.one_hot(proposal_panel.y,start.shape[1]).to(logits)-logits.softmax(dim=1)
    per_client=[]
    for j in range(len(proposal_panel.sizes)):
        ids=proposal_panel.groups==j
        gradient=(proposal_panel.x[ids].T@residual[ids])/proposal_panel.sizes[j]
        per_client.append(updates@gradient.flatten())
    sizes=proposal_panel.sizes.to(updates.dtype)
    gains=torch.stack(per_client).T@(sizes/sizes.sum())
    trusted=trust_weights(prior,gains)
    calibrated=updates.T.mv(trusted).reshape_as(start)
    reference=updates.T.mv(prior).reshape_as(start)
    models=[start+scale*torch.lerp(calibrated,reference,mix) for mix in (0.,.5,1.)]
    return models,[selection_panel.reports(m) for m in models]
