import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "watch_train.py"
SPEC = importlib.util.spec_from_file_location("watch_train", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
watch_train = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(watch_train)


def _write_config(tmp_path: Path, checkpoint: dict) -> Path:
    output_dir = tmp_path / "output"
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    torch.save(checkpoint, checkpoint_dir / "latest.ckpt")
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        yaml.safe_dump({"output_dir": str(output_dir), "resume": True}),
        encoding="utf-8",
    )
    return config_path


def test_preflight_validates_checkpoint_and_gpu_memory(monkeypatch, tmp_path) -> None:
    checkpoint = {key: {} for key in watch_train.CHECKPOINT_KEYS}
    checkpoint["epoch"] = 1200
    checkpoint["global_step"] = 288000
    config_path = _write_config(tmp_path, checkpoint)
    monkeypatch.setattr(watch_train, "_gpu_free_memory", lambda: {0: 45000, 1: 32000})

    summary = watch_train.preflight(config_path, (0, 1), 8192)

    assert "epoch=1200 global_step=288000" in summary
    assert "gpu0=45000MiB" in summary


def test_preflight_rejects_incomplete_checkpoint(monkeypatch, tmp_path) -> None:
    config_path = _write_config(tmp_path, {"epoch": 1})
    monkeypatch.setattr(watch_train, "_gpu_free_memory", lambda: {0: 45000, 1: 32000})

    try:
        watch_train.preflight(config_path, (0, 1), 8192)
    except ValueError as error:
        assert "missing keys" in str(error)
    else:
        raise AssertionError("an incomplete checkpoint must fail preflight")


def test_preflight_waits_for_enough_gpu_memory(monkeypatch, tmp_path) -> None:
    checkpoint = {key: {} for key in watch_train.CHECKPOINT_KEYS}
    checkpoint["epoch"] = 1
    checkpoint["global_step"] = 10
    config_path = _write_config(tmp_path, checkpoint)
    monkeypatch.setattr(watch_train, "_gpu_free_memory", lambda: {0: 45000, 1: 4096})

    try:
        watch_train.preflight(config_path, (0, 1), 8192)
    except RuntimeError as error:
        assert "insufficient free GPU memory" in str(error)
    else:
        raise AssertionError("a low-memory GPU must fail preflight")


def test_monitor_restarts_after_failure(monkeypatch) -> None:
    return_codes = iter([1, 0])
    children = []

    class _Child:
        pid = 12345

        def __init__(self):
            self.return_code = next(return_codes)

        def wait(self):
            return self.return_code

        def poll(self):
            return self.return_code

    def popen(*args, **kwargs):
        child = _Child()
        children.append(child)
        return child

    monkeypatch.setattr(
        watch_train,
        "parse_args",
        lambda: SimpleNamespace(
            config="configs/train.yaml",
            cuda_visible_devices="0,1",
            restart_delay=0.0,
            min_free_memory_mib=8192,
            check_only=False,
        ),
    )
    monkeypatch.setattr(watch_train, "preflight", lambda *args: "checks passed")
    monkeypatch.setattr(watch_train.subprocess, "Popen", popen)
    monkeypatch.setattr(watch_train.signal, "signal", lambda *args: None)

    assert watch_train.main() == 0
    assert len(children) == 2
