from types import SimpleNamespace

import numpy as np
import torch

from dexcg.models.contact.grammar import ContactGrammarLogitsProcessor
from dexcg.models.contact.planner import ContactPlanner


def test_contact_grammar_enforces_link_xyz_groups() -> None:
    tokenizer = SimpleNamespace(
        link_token_ids=(10, 11),
        position_token_ids=np.asarray((20, 21, 22)),
        joint_start_id=2,
        joint_end_id=3,
    )
    grammar = ContactGrammarLogitsProcessor(tokenizer)
    scores = torch.zeros(1, 32)

    after_start = grammar(torch.tensor([[2]]), scores.clone())
    assert after_start[0, 10] == 0
    assert after_start[0, 3] == 0
    assert after_start[0, 20] < -1e20

    after_link = grammar(torch.tensor([[2, 10]]), scores.clone())
    assert after_link[0, 20] == 0
    assert after_link[0, 11] < -1e20

    after_xyz = grammar(torch.tensor([[2, 10, 20, 21, 22]]), scores.clone())
    assert after_xyz[0, 11] == 0
    assert after_xyz[0, 3] == 0

    full_graph = torch.tensor([[2] + [10, 20, 21, 22] * 2])
    after_maximum = grammar(full_graph, scores.clone())
    assert after_maximum[0, 3] == 0
    assert after_maximum[0, 10] < -1e20


def test_planner_attention_bias_matches_llm_dtype() -> None:
    padding = torch.ones(1, 4, dtype=torch.long)
    groups = torch.tensor([[1, 0, 0, 1]])
    mask = ContactPlanner._attention_mask(padding, groups, torch.bfloat16)
    assert mask.dtype == torch.bfloat16
    assert mask.shape == (1, 1, 4, 4)
