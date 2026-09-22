import pytest

from dexcg.models.contact.partfield.encoder import PartFieldConfig
from dexcg.models.contact.projector import PointCloudProjector


def test_planner_dropout_defaults_are_disabled() -> None:
    config = PartFieldConfig()
    projector = PointCloudProjector(input_dim=4, output_dim=8)

    assert config.attention_dropout == 0.0
    assert config.mlp_dropout == 0.0
    assert projector.dropout.p == 0.0


def test_projector_dropout_is_configurable() -> None:
    projector = PointCloudProjector(input_dim=4, output_dim=8, dropout=0.05)
    assert projector.dropout.p == pytest.approx(0.05)
