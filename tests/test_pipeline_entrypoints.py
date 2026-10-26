from pathlib import Path
import json
import os
import shutil
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "name", ["run_generate_data.py", "run_train.sh", "model/download_basemodel.py"]
)
def test_documented_pipeline_entrypoints_are_bundled(name):
    assert (Path(__file__).resolve().parents[1] / name).is_file(), name


def test_readme_does_not_require_an_external_checkout():
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()
    assert "REFERENCE_ROOT" not in readme


@pytest.mark.parametrize("gpus", ["1", "4"])
def test_initial_training_launcher_runs_all_documented_stages(tmp_path, gpus):
    root = Path(__file__).resolve().parents[1]
    shutil.copyfile(root / "run_train.sh", tmp_path / "run_train.sh")
    mock = tmp_path / "mock_python"
    mock.write_text(
        f"#!{sys.executable}\n"
        + """
import json
import os
import sys
with open(os.environ['COMMAND_LOG'], 'a') as stream:
    stream.write(json.dumps(sys.argv[1:]) + '\\n')
"""
    )
    mock.chmod(0o755)
    log = tmp_path / "commands.jsonl"
    result = subprocess.run(
        [
            "bash",
            str(tmp_path / "run_train.sh"),
            "--stage",
            "all",
            "--category",
            "Beauty",
            "--gpus",
            gpus,
        ],
        cwd=tmp_path,
        env={
            **os.environ,
            "PYTHON": str(mock),
            "COMMAND_LOG": str(log),
            "USE_DEEPSPEED": "1",
            "DYNR2_ARTIFACT_ROOT": str(tmp_path / "artifacts"),
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert [call[call.index("--stage") + 1] for call in calls] == [
        "expand",
        "align",
        "merge",
        "rec",
        "sid_routing",
    ]
    for call in calls:
        assert call[call.index("--category") + 1] == "Beauty"
        phase = call[call.index("--stage") + 1]
        if gpus == "4" and phase not in ("expand", "merge"):
            assert call[:2] == ["-m", "torch.distributed.run"]
            assert call[call.index("--deepspeed") + 1] == "configs/zero2.json"
        else:
            assert call[:2] == ["-m", "dynr2.initialize"]
