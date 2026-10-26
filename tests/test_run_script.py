import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.mark.parametrize("domain", ["Beauty", "Sports", "Toys"])
@pytest.mark.parametrize("stage", ["all", "train", "test"])
def test_dataset_runner_selects_checkpoint_and_evaluates(tmp_path, domain, stage):
    root = Path(__file__).resolve().parents[1]
    (tmp_path / "scripts").mkdir()
    shutil.copyfile(root / "scripts" / "run.sh", tmp_path / "scripts" / "run.sh")
    storage = tmp_path / "artifacts"
    (storage / "models" / domain / "initial").mkdir(parents=True)
    data = tmp_path / "data" / domain
    data.mkdir(parents=True)
    for name in (
        "item_mapping.json",
        "train.parquet",
        "model_selection.parquet",
        "test.parquet",
        "train_history_counts.json",
    ):
        (data / name).touch()
    name = domain.lower()
    if stage == "train":
        (data / "test.parquet").unlink()
    elif stage == "test":
        (data / "train.parquet").unlink()
        (data / "model_selection.parquet").unlink()
        training = storage / "outputs" / f"{name}_train"
        (training / "epoch_2").mkdir(parents=True)
        (training / "training.json").write_text(json.dumps({"selected_adapter": "epoch_2"}))
    mock = tmp_path / "mock_python"
    mock.write_text(
        f"#!{sys.executable}\n"
        + """
import json
import os
from pathlib import Path
import sys

args = sys.argv[1:]
if args[0] == '-c':
    os.execv(sys.executable, [sys.executable, *args])
with open(os.environ['COMMAND_LOG'], 'a') as stream:
    stream.write(json.dumps(args) + '\\n')
output = Path(args[args.index('--output') + 1])
output.mkdir(parents=True)
if 'dynr2.train' in args:
    (output / 'epoch_2').mkdir()
    (output / 'training.json').write_text(json.dumps({'selected_adapter': 'epoch_2'}))
else:
    (output / 'metrics.json').write_text('{}')
"""
    )
    mock.chmod(0o755)
    log = tmp_path / "commands.jsonl"
    command = ["bash", str(tmp_path / "scripts" / "run.sh"), domain]
    if stage != "all":
        command.append(stage)
    result = subprocess.run(
        command,
        cwd=tmp_path,
        env={
            **os.environ,
            "PYTHON": str(mock),
            "GPUS": "4",
            "COMMAND_LOG": str(log),
            "DYNR2_ARTIFACT_ROOT": str(storage),
        },
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(calls) == (2 if stage == "all" else 1)
    if stage != "test":
        train = calls[0]
        assert train[:3] == ["-m", "torch.distributed.run", "--standalone"]
        assert "--nproc_per_node=4" in train
        assert train[train.index("--module") + 1] == "dynr2.train"
        assert train[train.index("--config") + 1] == f"configs/{name}.json"
        assert train[train.index("--model") + 1] == str(storage / "models" / domain / "initial")
    if stage != "train":
        evaluate = calls[-1]
        assert evaluate[:2] == ["-m", "dynr2.evaluate"]
        assert evaluate[evaluate.index("--adapter") + 1] == str(
            storage / "outputs" / f"{name}_train/epoch_2"
        )
        assert evaluate[evaluate.index("--data") + 1] == f"data/{domain}/test.parquet"
        assert (storage / "outputs" / f"{name}_test" / "metrics.json").is_file()
    else:
        assert not (storage / "outputs" / f"{name}_test").exists()
