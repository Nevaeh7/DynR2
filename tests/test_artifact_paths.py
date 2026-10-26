import pytest

from dynr2 import paths
from dynr2.runtime import save_response_adapter


def test_default_artifacts_are_a_sibling_of_the_repository(tmp_path, monkeypatch):
    repository = tmp_path / "DynR2"
    repository.mkdir()
    monkeypatch.setattr(paths, "REPOSITORY_ROOT", repository)
    monkeypatch.delenv("DYNR2_ARTIFACT_ROOT", raising=False)
    assert paths.artifact_root() == tmp_path / "DynR2-artifacts"


def test_model_writes_inside_repository_are_rejected(tmp_path, monkeypatch):
    repository = tmp_path / "repository"
    repository.mkdir()
    monkeypatch.setattr(paths, "REPOSITORY_ROOT", repository)
    target = repository / "outputs/epoch_1"
    with pytest.raises(ValueError, match="outside the repository"):
        save_response_adapter(None, target)
    assert not target.exists()


def test_artifact_override_must_stay_outside_repository(tmp_path, monkeypatch):
    repository = tmp_path / "repository"
    repository.mkdir()
    monkeypatch.setattr(paths, "REPOSITORY_ROOT", repository)
    monkeypatch.setenv("DYNR2_ARTIFACT_ROOT", str(repository / "checkpoints"))
    with pytest.raises(ValueError, match="outside the repository"):
        paths.artifact_root()
    monkeypatch.setenv("DYNR2_ARTIFACT_ROOT", str(tmp_path / "storage"))
    assert paths.artifact_root() == tmp_path / "storage"
