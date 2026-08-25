import pytest

from benchmarks.checkpoint_store import SQLiteCheckpointStore


def _ok_row(run_key: str, prediction: str = "4"):
    return {
        "status": "ok",
        "run_key": run_key,
        "prediction": prediction,
    }


def test_sqlite_checkpoint_round_trip_and_complete_state(tmp_path):
    path = tmp_path / "grid.sqlite"
    store = SQLiteCheckpointStore(path, fingerprint="f" * 64, resume=False)
    store.append_run(_ok_row("a" * 64))
    store.upsert_parity(
        {"model": "m", "dataset": "d", "sample_id": "s", "token_match": True}
    )
    store.mark_complete()
    store.close()

    resumed = SQLiteCheckpointStore(path, fingerprint="f" * 64, resume=True)
    payload = resumed.load()
    resumed.close()

    assert payload["state"] == "complete"
    assert payload["runs"] == [_ok_row("a" * 64)]
    assert payload["parity_records"][0]["token_match"] is True


def test_sqlite_checkpoint_rejects_mismatch_and_conflicting_duplicate(tmp_path):
    path = tmp_path / "grid.sqlite"
    store = SQLiteCheckpointStore(path, fingerprint="f" * 64, resume=False)
    store.append_run(_ok_row("a" * 64))
    with pytest.raises(ValueError, match="Conflicting"):
        store.append_run(_ok_row("a" * 64, prediction="5"))
    store.close()

    with pytest.raises(ValueError, match="does not match"):
        SQLiteCheckpointStore(path, fingerprint="e" * 64, resume=True)


def test_sqlite_checkpoint_requires_explicit_resume_to_preserve_data(tmp_path):
    path = tmp_path / "grid.sqlite"
    store = SQLiteCheckpointStore(path, fingerprint="f" * 64, resume=False)
    store.close()

    with pytest.raises(FileExistsError, match="use resume"):
        SQLiteCheckpointStore(path, fingerprint="f" * 64, resume=False)
