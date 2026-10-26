"""Keep generated models and checkpoints outside the source repository."""

import os
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def external_checkpoint_path(path):
    path = Path(path).expanduser().absolute()
    repository = REPOSITORY_ROOT.resolve()
    if path.is_relative_to(repository) or path.resolve().is_relative_to(repository):
        raise ValueError("Model weights and checkpoints must be saved outside the repository")
    return path


def artifact_root(project_root=None):
    project = Path(project_root or REPOSITORY_ROOT).resolve()
    configured = os.environ.get("DYNR2_ARTIFACT_ROOT")
    if configured:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            path = project / path
    else:
        path = project.parent / (project.name + "-artifacts")
    return external_checkpoint_path(path)
