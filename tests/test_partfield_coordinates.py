import torch

from dexcg.models.contact.partfield.modules.PVCNN.encoder_pc import coordinate2index


def test_coordinate2index_clamps_rounded_low_precision_boundaries() -> None:
    coordinates = torch.tensor(
        [[[-0.1, 0.0], [1.0 - 1.0e-5, 1.1]]], dtype=torch.bfloat16
    )

    indices = coordinate2index(coordinates, resolution=256)

    assert indices.tolist() == [[[0, 65535]]]
