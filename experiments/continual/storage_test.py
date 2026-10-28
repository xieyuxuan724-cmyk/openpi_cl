import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest import mock

import pytest

from experiments.continual.storage import check_space
from experiments.continual.storage import configure


@mock.patch.dict(os.environ)
def test_paths_and_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", tempfile.tempdir)
    monkeypatch.setattr(sys, "dont_write_bytecode", sys.dont_write_bytecode)
    root = tmp_path / "runtime"
    output = root / "checkpoints/seq42"
    result = configure(root, output, minimum_gib=0, require_data_disk=False)
    assert not output.exists()
    assert Path(os.environ["TMPDIR"]).is_relative_to(root)
    assert Path(os.environ["JAX_COMPILATION_CACHE_DIR"]).is_relative_to(root)
    # A fresh interpreter must see tmp/cache settings before third-party imports.
    actual = subprocess.check_output(
        [sys.executable, "-c", "import tempfile; print(tempfile.gettempdir())"], text=True
    ).strip()
    assert actual == str(root / "tmp")
    assert result["output"] == str(output)


def test_reject_system_disk_and_output_escape(tmp_path):
    with (
        mock.patch.object(Path, "stat", return_value=mock.Mock(st_dev=1)),
        pytest.raises(ValueError, match="system filesystem"),
    ):
        configure(tmp_path)
    with pytest.raises(ValueError, match="Checkpoint output"):
        configure(tmp_path / "runtime", tmp_path / "outside", minimum_gib=0, require_data_disk=False)


def test_reject_symlink_escape(tmp_path):
    root = tmp_path / "runtime"
    root.mkdir()
    (root / "cache").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        configure(root, minimum_gib=0, require_data_disk=False)


def test_space_guard():
    with (
        mock.patch("shutil.disk_usage", return_value=mock.Mock(free=2**30)),
        pytest.raises(OSError, match="Insufficient"),
    ):
        check_space(Path("."), 2)


def test_trainer_rejects_unmanaged_cl_paths(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from scripts import train

    config = SimpleNamespace(name="cl_storage_guard", checkpoint_dir=tmp_path / "outside")
    monkeypatch.delenv("OPENPI_CL_STORAGE_ROOT", raising=False)
    with pytest.raises(ValueError, match="storage launcher"):
        train.main(config)
    monkeypatch.setenv("OPENPI_CL_STORAGE_ROOT", str(tmp_path / "runtime"))
    with pytest.raises(ValueError, match="checkpoints must be under"):
        train.main(config)
