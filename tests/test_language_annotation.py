import json

import pytest

from dexcg.annotation.language import parse_annotation


ANNOTATION = {
    "class_name": "faucet",
    "grasped_object_part": "handle",
    "low_level_grasp_instruction": "Grasp the handle with the fingertips.",
    "high_level_grasp_instruction": "Turn the handle to open the faucet.",
}


def test_parse_plain_json_annotation() -> None:
    assert parse_annotation(json.dumps(ANNOTATION)) == ANNOTATION


def test_parse_json_code_fence_added_by_gemma() -> None:
    raw = f"```json\n{json.dumps(ANNOTATION)}\n```"

    assert parse_annotation(raw) == ANNOTATION


def test_reject_annotation_with_unexpected_fields() -> None:
    with pytest.raises(ValueError, match="Unexpected annotation fields"):
        parse_annotation('{"class_name": "faucet"}')


def test_invalid_json_error_includes_model_output() -> None:
    with pytest.raises(ValueError, match="not-json"):
        parse_annotation("```json\nnot-json\n```")
