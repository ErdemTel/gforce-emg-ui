"""Live Classifier with a channel-subset, two-class model (low / high, no ch6).

Real-model tests use emg-stiffness-data runs/mlp_s21_7ch_binary_seed42_001 (30 training
windows per trial) and runs/mlp_s21_7ch_binary_155w_seed42_001 (155) with their training
trials in data/S2.1 (skipped when not available).
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from gforce_emg_ui.live.engine import HUB_CHANNELS
from gforce_emg_ui.live.model import LoadedModel, ModelError, classifier_status, load_trusted_model
from gforce_emg_ui.live.stabilization import Observation, Stabilizer, StabilizerSettings
from test_live_model import INSTALLED_CODE_SHA256, run_live

REPO = Path(__file__).resolve().parents[1]
BINARY = Path(os.environ.get(
    "GFORCE_LIVE_BINARY_MODEL",
    REPO.parent / "emg-stiffness-data/runs/mlp_s21_7ch_binary_seed42_001/model.joblib"))
BINARY_DATA = Path(os.environ.get("GFORCE_LIVE_BINARY_DATA",
                                  REPO.parent / "emg-stiffness-data/data/S2.1"))
BINARY_SHA256 = "1e54356b45d4248828c2eb467c483ddbe8927a8bd5d4d2f34d0d4b08ac03435c"
# name -> (model.joblib, sha256, training windows); the 155-window model uses the whole HOLD core
BINARY_MODELS = {
    "30w": (BINARY, BINARY_SHA256, 600),
    "155w": (BINARY.parents[1] / "mlp_s21_7ch_binary_155w_seed42_001/model.joblib",
             "359c91d5679624f96f8f865fd263b0726efb244395d75d83b93d814b7ebbef90", 3100)}
KEEP = ("ch1", "ch2", "ch3", "ch4", "ch5", "ch7", "ch8")
BINARY_CLASSES = ((0, "low", 0.0), (2, "high", 1.0))

needs_binary = pytest.mark.skipif(not (classifier_status()[0] and BINARY.is_file()),
                                  reason="emg-classifier or the binary model not available")
needs_binary_data = pytest.mark.skipif(not BINARY_DATA.is_dir(), reason="S2.1 not available")


@pytest.fixture(scope="module", params=list(BINARY_MODELS))
def binary_case(request):
    """(loaded model, expected sha256, expected training windows)."""
    path, sha, n_windows = BINARY_MODELS[request.param]
    if not (classifier_status()[0] and path.is_file()):
        pytest.skip(f"emg-classifier or {path.parent.name} not available")
    return load_trusted_model(path), sha, n_windows


@pytest.fixture(scope="module")
def binary(binary_case):
    return binary_case[0]


def test_binary_artifact_is_accepted(binary_case):
    binary, sha, n_windows = binary_case
    assert binary.sha256 == sha and binary.name == "short_window_mlp"
    assert binary.channel_names == KEEP and binary.classes == BINARY_CLASSES
    assert (binary.window_samples, binary.hop_samples) == (100, 25)
    assert list(binary.clf.pipeline.classes_) == [0, 2]
    assert binary.training["n_trials"] == 20 and binary.training["n_windows"] == n_windows
    assert binary.code_sha256 == binary.installed_code_sha256 == INSTALLED_CODE_SHA256
    assert binary.warnings == ()


@needs_binary_data
def test_binary_live_adapter_matches_offline_api(binary):
    """8-channel live blocks -> the engine's 7 model channels -> same windows, features and
    probabilities as the offline API on the 7-channel HOLD windows; never 'medium'."""
    from emg_classifier.trials import load_core, read_trial
    from emg_classifier.windows import make_windows, stack
    idx = [HUB_CHANNELS.index(c) for c in binary.channel_names]
    n = 0
    for t in binary.training["trials"]:
        rec = read_trial(BINARY_DATA / t["folder"])
        core, _ = load_core(rec, binary.hold_rule["start_margin_s"],
                            binary.hold_rule["end_margin_s"])
        wins = make_windows(core, binary.window_samples, binary.hop_samples, None)
        X = list(stack(core, wins.starts, binary.window_samples)[..., idx])
        off = binary.clf.predict_windows(X, channel_names=binary.channel_names)
        rec_clf, rows, snap = run_live(binary, core, sizes=(16, 37))
        assert snap.counters["resets"] == {} and len(rec_clf.windows) == len(X) == len(rows)
        for lw, ow in zip(rec_clf.windows, X):
            assert lw.shape == (100, 7)
            np.testing.assert_array_equal(lw, ow)
        for r, p in zip(rows, off):
            assert r["valid"] == 1 and r["class_id"] == p.class_id in (0, 2)
            assert r["label"] in ("low", "high") and r["p_medium"] is None
            assert r["s_discrete"] == p.s_discrete in (0.0, 1.0)
            np.testing.assert_allclose([r["p_low"], r["p_high"]],
                                       [p.probabilities["low"], p.probabilities["high"]],
                                       rtol=1e-12, atol=1e-15)
        n += len(rows)
    assert n > 2500


def _artifact(tmp_path, edit):
    import joblib
    obj = joblib.load(BINARY)
    obj = {**obj, "metadata": json.loads(json.dumps(obj["metadata"]))}
    edit(obj["metadata"])
    path = tmp_path / "model.joblib"
    joblib.dump(obj, path)
    (tmp_path / "model.json").write_text(json.dumps(obj["metadata"]))
    return path


@needs_binary
@pytest.mark.parametrize("edit,match", [
    (lambda m: m.update(channel_names=list(KEEP[::-1])), "ordered subset"),
    (lambda m: m.update(channel_names=list(KEEP[:6]) + ["ch9"]), "ordered subset"),
    (lambda m: m.update(classes=m["classes"][:1]), "at least two"),
    (lambda m: m["classes"][1].update(label="medium", class_id=1, s=0.5), "classes"),
])
def test_bad_subset_metadata_is_rejected(tmp_path, edit, match):
    with pytest.raises(ModelError, match=match):
        load_trusted_model(_artifact(tmp_path, edit))


# -- stabilizer with two classes --------------------------------------------------------

def _ob(pid, label, t, probs):
    return Observation(pid, "r", 1, 1, 0, 100 + 25 * pid, 25, t, t + 0.002, True, label, probs)


def test_stabilizer_accepts_two_class_predictions():
    st = Stabilizer(StabilizerSettings(enabled=True))
    pid, t = 0, 1.0
    for label in ("low",) * 4 + ("high",) * 4:
        pid += 1
        t += 0.05
        probs = {"low": 0.9, "high": 0.1} if label == "low" else {"low": 0.1, "high": 0.9}
        row = st.observe(_ob(pid, label, t, probs), now=t + 0.003)
        assert row["reason"] != "invalid_prediction" and row["p_medium"] is None
    assert st.state().accepted == "high" and st.state().s == 1.0
    for probs, label in (({"low": 0.9, "high": 0.1}, "medium"),      # label not in the scores
                         ({"low": 0.5, "max": 0.5}, "low"),          # unknown label
                         ({"low": 1.0}, "low"),                      # one class only
                         ({"low": float("nan"), "high": 0.5}, "high")):
        pid += 1
        assert st.observe(_ob(pid, label, t, probs), now=t)["reason"] == "invalid_prediction"


# -- dashboard follows the model's classes -----------------------------------------------

def _fake(dash, channels, classes, tmp_path):
    from gforce_emg_ui.live import compat
    from test_live_engine import FakeClf
    acq = compat.live_acquisition(dash._device, dash._params, dash.source, dash.hub)
    return LoadedModel(
        clf=FakeClf(), path=tmp_path / "model.joblib", sha256="0" * 64, sidecar_sha256=None,
        name="short_window_mlp", window_samples=100, hop_samples=25, window_units="samples",
        channel_names=channels, classes=classes, feature={"name": "log_ac_rms", "version": 1},
        acquisition={"keys": ["live"], "configurations": {"live": {"summary": acq}}},
        training={"n_trials": 20, "n_windows": 600}, cv_reference={}, hold_rule={}, created="",
        code_sha256=None, installed_code_sha256="", trained_versions={}, installed_versions={},
        limitations=(), warnings=())


def test_dashboard_shows_only_the_models_classes(qapp, tmp_path):
    from test_live_stabilization import make_dash
    from gforce_emg_ui.ui.live_view import ProbBars
    dash, _ = make_dash(tmp_path)
    ticks = lambda: [t for _, t in dash.plot.getAxis("left")._tickLevels[0]]   # noqa: E731
    try:
        assert dash.bars.labels == ("low", "medium", "high")           # before any model
        dash._load_token += 1
        dash._on_model_loaded(_fake(dash, KEEP, BINARY_CLASSES, tmp_path), dash._load_token)
        assert dash.model is not None and dash.bars.labels == ("low", "high")
        assert dash.bars.height() == ProbBars.ROW * 2 + 8
        assert ticks() == ["0 low", "1 high"]
        assert dash._soft_formula == "0·p_low + 1·p_high"
        assert "not used: ch6" in dash.model_label.text()
        dash.bars.set_probabilities({"low": 0.2, "high": 0.8})
        assert dash.bars.values() == (0.2, 0.8)
        dash._load_token += 1
        three = ((0, "low", 0.0), (1, "medium", 0.5), (2, "high", 1.0))
        dash._on_model_loaded(_fake(dash, HUB_CHANNELS, three, tmp_path), dash._load_token)
        assert dash.bars.labels == ("low", "medium", "high") and dash.bars.values() is None
        assert ticks() == ["0 low", "0.5 medium", "1 high"]
        assert "not used" not in dash.model_label.text()
    finally:
        dash.shutdown()
