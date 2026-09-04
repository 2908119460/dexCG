import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "collect_dexart.py"
SPEC = importlib.util.spec_from_file_location("collect_dexart", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
collect_dexart = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collect_dexart)


def test_balanced_collection_settings_uses_split_and_quota(monkeypatch) -> None:
    monkeypatch.setattr(collect_dexart, "split_object_ids", lambda task, split: [10, 20])

    settings = collect_dexart.balanced_collection_settings(
        "faucet",
        {"quota_per_object": 14},
        {"split": "seen", "max_attempts_per_object": 200},
    )

    assert settings == ([10, 20], 14, 200)


@pytest.mark.parametrize(
    ("quota", "attempts"),
    [(0, 200), (-1, 200), (14, 13), (14, None)],
)
def test_balanced_collection_settings_rejects_invalid_limits(
    monkeypatch, quota, attempts
) -> None:
    monkeypatch.setattr(collect_dexart, "split_object_ids", lambda task, split: [10])

    with pytest.raises(ValueError):
        collect_dexart.balanced_collection_settings(
            "faucet",
            {"quota_per_object": quota},
            {"split": "seen", "max_attempts_per_object": attempts},
        )


def test_empty_object_statistics_tracks_quota_per_object() -> None:
    statistics = collect_dexart.empty_object_statistics([148, 693], 14)

    assert list(statistics) == ["148", "693"]
    assert statistics["148"]["quota"] == 14
    assert statistics["148"]["attempts"] == 0
    assert statistics["148"]["episode_indices"] == []
