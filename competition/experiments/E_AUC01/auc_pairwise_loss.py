"""Pairwise logistic ranking on raw logits, never sigmoid probabilities."""
import torch
from torch.nn import functional as F


def pairwise_auc_loss(logits, labels):
    positive = logits[labels == 1].float()
    negative = logits[labels == 0].float()
    if not len(positive) or not len(negative):
        raise ValueError('Ranking batch must contain both classes')
    return F.softplus(negative[:, None] - positive[None, :]).mean()


def check_loss():
    labels = torch.tensor([1., 1., 0., 0.])
    good = torch.tensor([2., 3., -2., -3.], requires_grad=True)
    bad = torch.tensor([-2., -3., 2., 3.], requires_grad=True)
    good_loss = pairwise_auc_loss(good, labels)
    bad_loss = pairwise_auc_loss(bad, labels)
    assert good_loss < bad_loss
    bad_loss.backward()
    assert (bad.grad[:2] < 0).all() and (bad.grad[2:] > 0).all()
    extreme = torch.tensor([-10000., 10000.], requires_grad=True)
    extreme_loss = pairwise_auc_loss(extreme, torch.tensor([1., 0.]))
    extreme_loss.backward()
    assert torch.isfinite(extreme_loss) and torch.isfinite(extreme.grad).all()
    try:
        pairwise_auc_loss(torch.ones(4), torch.ones(4))
    except ValueError:
        pass
    else:
        raise AssertionError('Single-class ranking batch was accepted')
    return {'direction_ok': True, 'gradient_sign_ok': True, 'extreme_logits_finite': True,
            'missing_class_rejected': True, 'good_loss': float(good_loss.detach()),
            'bad_loss': float(bad_loss.detach())}


if __name__ == '__main__':
    print(check_loss())
