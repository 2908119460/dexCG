import torch

from dexcg.common.typing import ContactPlan
from dexcg.policy.contact_cache import ContactCache


def test_contact_cache_updates_at_slow_rate_and_on_language_change() -> None:
    cache = ContactCache(update_interval=3)
    calls = 0

    def planner() -> ContactPlan:
        nonlocal calls
        calls += 1
        return ContactPlan(torch.tensor([[calls]]), torch.ones(1, 1, dtype=torch.bool))

    assert cache.get(0, ["grasp cup"], planner).token_ids.item() == 1
    assert cache.get(1, ["grasp cup"], planner).token_ids.item() == 1
    assert cache.get(2, ["grasp cup"], planner).token_ids.item() == 1
    assert cache.get(3, ["grasp cup"], planner).token_ids.item() == 2
    assert cache.get(4, ["lift cup"], planner).token_ids.item() == 3
