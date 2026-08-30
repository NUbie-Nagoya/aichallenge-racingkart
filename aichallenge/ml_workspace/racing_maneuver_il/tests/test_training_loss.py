import torch

from racing_maneuver_il.train import grouped_action_change_loss


def test_action_change_loss_ignores_batch_group_boundaries_and_gaps():
    prediction = torch.tensor([[0.0, 0.0], [1.0, 1.0], [10.0, 10.0], [12.0, 12.0]])
    target = torch.tensor([[0.0, 0.0], [0.0, 0.0], [10.0, 10.0], [11.0, 11.0]])
    groups = ["a", "a", "b", "b"]
    indices = torch.tensor([5, 6, 20, 22])

    loss = grouped_action_change_loss(prediction, target, groups, indices)

    # Only rows 0->1 are chronological in the same group. Smooth-L1(1, 0)=0.5.
    assert float(loss) == 0.5


def test_action_change_loss_is_zero_when_no_valid_pair():
    prediction = torch.tensor([[0.0, 0.0], [3.0, 3.0]])
    target = torch.zeros_like(prediction)
    loss = grouped_action_change_loss(
        prediction, target, ["a", "b"], torch.tensor([1, 2])
    )
    assert float(loss) == 0.0
