"""Live Classifier with the REAL selected model (skipped when the model, the recordings or
emg-classifier are not available). Parity compares the live adapter with the frozen
model's own offline API on identical recorded HOLD windows: implementation parity on
training trials, not a generalization result."""

import dataclasses
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from gforce_emg_ui.hub import DataHub, EmgBlock
from gforce_emg_ui.live.engine import INVALID_INPUT, RUNNING, LiveEngine, LiveSettings
from gforce_emg_ui.live.model import ModelError, classifier_status, load_trusted_model

REPO = Path(__file__).resolve().parents[1]
MODEL = Path(os.environ.get("GFORCE_LIVE_MODEL",
                            REPO.parent / "emg-stiffness-data/runs/lda_hold_001/model.joblib"))
DATA = Path(os.environ.get("GFORCE_LIVE_DATA", REPO.parent / "emg-stiffness-data/Erdem"))
MODEL_SHA256 = "54f12a4a4a17d38f71380518a0bb12876df9a975de234178fd0a4caade16b8c4"
CODE_SHA256 = "aaf3646bb71753e5e24e72e47e54f75ea4f89ba66e551cb30a236978b19836c6"

needs_model = pytest.mark.skipif(not (classifier_status()[0] and MODEL.is_file()),
                                 reason="emg-classifier or the selected model not available")
needs_data = pytest.mark.skipif(not DATA.is_dir(), reason="recorded trials not available")


class Clock:
    def __init__(self, t=5000.0):
        self.t = t

    def __call__(self):
        return self.t


class Recording:
    """Wraps the real classifier: records the exact windows sent, then delegates."""

    def __init__(self, clf):
        self.clf, self.windows = clf, []

    def predict_windows(self, windows, *, channel_names):
        self.windows += [np.array(w) for w in windows]
        return self.clf.predict_windows(windows, channel_names=channel_names)


class Capture:
    def __init__(self):
        self.rows, self.events, self.closed = [], [], False

    def put_prediction(self, row):
        self.rows.append(row)

    def put_event(self, row):
        self.events.append(row)

    def put_display(self, row):
        pass

    def close(self, final=None):
        self.closed = True


@pytest.fixture(scope="module")
def model():
    return load_trusted_model(MODEL)


def offline(model, folder):
    """The frozen model's offline path: HOLD core -> all usable windows -> predict_windows."""
    from emg_classifier.trials import load_core, read_trial
    from emg_classifier.windows import make_windows, stack
    rec = read_trial(folder)
    core, _ = load_core(rec, model.hold_rule["start_margin_s"], model.hold_rule["end_margin_s"])
    wins = make_windows(core, model.window_samples, model.hop_samples, None)
    assert not wins.unusable
    X = list(stack(core, wins.starts, model.window_samples))
    return rec, core, wins, X, model.clf.predict_windows(X, channel_names=model.channel_names)


def run_live(model, core, sizes=(16,), dtype=np.float64, clock=None):
    """Feed the HOLD core samples as live EMG blocks through the DataHub."""
    clock = clock or Clock()
    rec_clf = Recording(model.clf)
    m = dataclasses.replace(model, clf=rec_clf)
    hub = DataHub()
    eng = LiveEngine(hub, LiveSettings(watchdog_s=0, inbox_samples=10 ** 6), clock=clock)
    eng.set_model(m)
    cap = Capture()
    eng.start(log=cap)
    n, pos, k, pushed = len(core.data), 0, 0, 0
    while pos < n:
        b = min(sizes[k % len(sizes)], n - pos)
        idx = core.sample_index[pos:pos + b].astype(np.int64)
        hub.push_emg(EmgBlock(
            data=core.data[pos:pos + b].T.astype(dtype), valid=np.ones((8, b), bool),
            sample_index=idx, device_time=idx / 1000.0, lost=np.zeros(b, bool),
            host_time=1.79e9, host_mono=clock.t, sample_rate=1000.0, stream_id=1))
        pos += b
        k += 1
        pushed += 1
    end = time.monotonic() + 20
    while time.monotonic() < end:
        c = eng.snapshot().counters
        if c.get("blocks", 0) >= pushed:
            break
        time.sleep(0.005)
    snap = eng.snapshot()
    eng.stop()
    eng.shutdown()
    return rec_clf, cap.rows, snap


# -- identity of the selected artifact ------------------------------------------------

@needs_model
def test_selected_artifact_is_the_final_short_window_model(model):
    assert model.sha256 == MODEL_SHA256 and model.name == "short_window_lda"
    assert (model.window_samples, model.hop_samples) == (100, 25)
    assert "samples" in model.window_units
    assert model.channel_names == tuple(f"ch{i}" for i in range(1, 9))
    assert model.classes == ((0, "low", 0.0), (1, "medium", 0.5), (2, "high", 1.0))
    assert (model.feature["name"], model.feature["version"]) == ("log_ac_rms", 1)
    steps = model.clf.pipeline.steps
    assert [type(e).__name__ for _, e in steps] == ["StandardScaler",
                                                    "LinearDiscriminantAnalysis"]
    lda = steps[1][1]
    assert (lda.solver, lda.shrinkage) == ("lsqr", "auto")
    assert list(model.clf.pipeline.classes_) == [0, 1, 2]
    # fitted on all 51 eligible trials x 10 windows, not a CV-fold model
    assert model.training["n_trials"] == 51 and model.training["n_windows"] == 510
    assert int(steps[0][1].n_samples_seen_) == 510
    # corrected rerun: the installed classifier sources are the ones it was trained with
    assert model.code_sha256 == model.installed_code_sha256 == CODE_SHA256
    assert model.warnings == ()


@needs_model
def test_scaler_matches_the_saved_training_windows(model):
    import csv
    run = MODEL.parent
    win = {r["window_id"]: r for r in csv.DictReader(open(run / "windows.csv"))}
    feat = {r["window_id"]: r for r in csv.DictReader(open(run / "features.csv"))}
    sel = [w for w, r in win.items() if r["selected_for_training"] == "True"]
    X = np.array([[float(feat[w][f"f_ch{c}"]) for c in range(1, 9)] for w in sel])
    sc = model.clf.pipeline.steps[0][1]
    assert len(sel) == 510 and len({w.split("#")[0] for w in sel}) == 51
    np.testing.assert_array_equal(X.mean(axis=0), sc.mean_)


# -- offline / live parity ---------------------------------------------------------------

@needs_model
@needs_data
def test_live_adapter_matches_offline_api_on_all_recorded_hold_windows(model):
    from emg_classifier import features as F
    from emg_classifier.trials import discover
    folders = discover(DATA)
    assert len(folders) == 51
    n_windows, max_diff, disagree, wrong = 0, 0.0, [], 0
    for folder in folders:
        rec, core, wins, X, off = offline(model, folder)
        rec_clf, rows, snap = run_live(model, core)
        assert snap.counters["resets"] == {} and snap.counters["dropped_blocks"] == 0
        assert len(rec_clf.windows) == len(X) == len(rows)
        for k, (lw, ow) in enumerate(zip(rec_clf.windows, X)):
            np.testing.assert_array_equal(lw, ow)                   # identical windows
            s = wins.starts[k]
            assert rows[k]["window_start_index"] == core.sample_index[s]
            assert rows[k]["window_end_index"] == core.sample_index[s + 99]
        np.testing.assert_array_equal(F.log_ac_rms(np.stack(rec_clf.windows), 1e-12),
                                      F.log_ac_rms(np.stack(X), 1e-12))   # same features
        for r, p in zip(rows, off):
            assert r["valid"] == 1 and r["class_id"] == p.class_id   # incl. matching mistakes
            live_p = np.array([r["p_low"], r["p_medium"], r["p_high"]])
            off_p = np.array([p.probabilities[l] for l in ("low", "medium", "high")])
            np.testing.assert_allclose(live_p, off_p, rtol=1e-12, atol=1e-15)
            max_diff = max(max_diff, float(np.abs(live_p - off_p).max()))
            assert r["s_discrete"] == p.s_discrete
        n_windows += len(rows)
        wrong += sum(r["label"] != rec.info["label"] for r in rows)   # same in both paths
        mean = np.mean([[p.probabilities[l] for l in ("low", "medium", "high")] for p in off], 0)
        if ("low", "medium", "high")[int(np.argmax(mean))] != rec.info["label"]:
            disagree.append(rec.trial_uid)
    print(f"\nparity: {n_windows} windows / {len(folders)} trials, max |dp| {max_diff:.2e}; "
          f"in-sample (training data) windows whose class differs from the instructed label "
          f"in BOTH paths: {wrong}; HOLD-mean disagreements: {disagree}")
    assert n_windows > 2500


@needs_model
@needs_data
@pytest.mark.parametrize("sizes", [(1,), (16,), (37,), (500,), (5, 16, 3, 250), (10 ** 6,)])
def test_real_windows_do_not_depend_on_block_size(model, sizes):
    from emg_classifier.trials import discover
    folder = [f for f in discover(DATA) if "_t005_" in f.name][0]   # an OOF-misclassified trial
    _, core, wins, X, off = offline(model, folder)
    _, rows, _ = run_live(model, core, sizes=sizes)
    assert [r["window_end_index"] for r in rows] == [int(core.sample_index[s + 99])
                                                      for s in wins.starts]
    assert [r["class_id"] for r in rows] == [p.class_id for p in off]


@needs_model
@needs_data
def test_float32_live_values_match_offline_within_rounding(model):
    """Live blocks are float32; emg.csv stores them with 9 significant digits."""
    from emg_classifier.trials import discover
    folder = [f for f in discover(DATA) if "_t189_" in f.name][0]
    _, core, _, X, off = offline(model, folder)
    np.testing.assert_array_equal(core.data.astype(np.float32).astype(np.float64)
                                  .astype(np.float32), core.data.astype(np.float32))
    _, rows, _ = run_live(model, core, dtype=np.float32)
    assert [r["class_id"] for r in rows] == [p.class_id for p in off]
    diff = max(abs(r[f"p_{l}"] - p.probabilities[l]) for r, p in zip(rows, off)
               for l in ("low", "medium", "high"))
    assert diff < 1e-6, diff


# -- frozen inference / invalid input -------------------------------------------------------

@needs_model
@needs_data
def test_inference_never_fits_or_changes_the_model(model, monkeypatch):
    import joblib
    from emg_classifier.trials import discover
    pipe = model.clf.pipeline

    def forbidden(*a, **k):
        raise AssertionError("fit/partial_fit/set_params called during live inference")
    for cls in {type(pipe), *(type(est) for _, est in pipe.steps)}:
        for name in ("fit", "partial_fit", "fit_transform", "set_params"):
            if hasattr(cls, name):
                monkeypatch.setattr(cls, name, forbidden)
    _, core, *_ = offline(model, discover(DATA)[0])
    _, rows, _ = run_live(model, core)
    monkeypatch.undo()
    assert rows
    # after every live run in this module: parameters equal the file's
    assert joblib.hash(pipe) == joblib.hash(joblib.load(MODEL)["pipeline"])


@needs_model
def test_stuck_or_invalid_input_is_never_s0(model):
    hub = DataHub()
    clock = Clock()
    eng = LiveEngine(hub, LiveSettings(watchdog_s=0), clock=clock)
    eng.set_model(model)
    eng.start()

    def push(start, data, valid=None):
        n = data.shape[1]
        idx = np.arange(start, start + n)
        hub.push_emg(EmgBlock(data=data.astype(np.float32),
                              valid=np.ones((8, n), bool) if valid is None else valid,
                              sample_index=idx, device_time=idx / 1000.0,
                              lost=np.zeros(n, bool), host_time=0.0, host_mono=clock.t,
                              sample_rate=1000.0, stream_id=1))
    for k in range(10):                                   # stuck band: all channels constant
        push(k * 16, np.full((8, 16), 3.0))
    deadline = time.monotonic() + 5
    while eng.snapshot().counters.get("blocks", 0) < 10 and time.monotonic() < deadline:
        time.sleep(0.005)
    s = eng.snapshot()
    assert s.state == INVALID_INPUT and "unusable_window" in s.reason
    assert s.current(clock.t) is None and s.result.prediction.s_discrete is None
    rng = np.random.default_rng(0)
    for k in range(10, 20):                               # real-looking signal -> valid
        push(k * 16, rng.normal(0, 5, (8, 16)))
    deadline = time.monotonic() + 5
    while eng.snapshot().state != RUNNING and time.monotonic() < deadline:
        time.sleep(0.005)
    assert eng.snapshot().current(clock.t).prediction.s_discrete in (0.0, 0.5, 1.0)
    bad = np.ones((8, 16), bool)
    bad[3] = False                                        # channel 4 missing
    push(20 * 16, rng.normal(0, 5, (8, 16)), valid=bad)
    deadline = time.monotonic() + 5
    while eng.snapshot().state != INVALID_INPUT and time.monotonic() < deadline:
        time.sleep(0.005)
    s = eng.snapshot()
    assert s.state == INVALID_INPUT and s.current(clock.t) is None and s.result is None
    eng.shutdown()


# -- loading errors -------------------------------------------------------------------------

def _artifact(tmp_path, edit=None, sidecar=True, name="model.joblib"):
    import joblib
    obj = joblib.load(MODEL)
    obj = {**obj, "metadata": json.loads(json.dumps(obj["metadata"]))}
    if edit:
        edit(obj)
    path = tmp_path / name
    joblib.dump(obj, path)
    if sidecar:
        (tmp_path / "model.json").write_text(json.dumps(obj["metadata"]))
    return path


@needs_model
@pytest.mark.parametrize("case,edit,match", [
    ("whole_hold", lambda o: o["metadata"].update(model_name="whole_hold_lda"), "whole-HOLD"),
    ("baseline", lambda o: o["metadata"].update(model_name="global_amplitude_lda"),
     "global-amplitude"),
    ("unknown_type", lambda o: o["metadata"].update(model_name="cnn"), "unknown model type"),
    ("artifact_version", lambda o: o.update(artifact_version=99), "artifact_version"),
    ("no_window", lambda o: o["metadata"].pop("window_samples"), "window"),
    ("ms_units", lambda o: o["metadata"].update(window_units="ms"), "not sample counts"),
    ("feature", lambda o: o["metadata"]["feature"].update(version=2), "feature"),
    ("classes", lambda o: o["metadata"]["classes"][0].update(s=0.25), "class mapping"),
    ("fold_model", lambda o: o["metadata"]["training_set"].update(n_windows=400, n_trials=40),
     "not the final model"),
    ("sklearn", lambda o: o["metadata"]["package_versions"].update(**{"scikit-learn": "0.1"}),
     "scikit-learn"),
    ("no_acquisition", lambda o: o["metadata"]["acquisition"].update(keys=[]), "acquisition"),
])
def test_unusable_artifacts_are_rejected_with_a_message(tmp_path, case, edit, match):
    with pytest.raises(ModelError, match=match):
        load_trusted_model(_artifact(tmp_path, edit))


@needs_model
def test_corrupt_missing_and_mismatched_files(tmp_path):
    bad = tmp_path / "corrupt.joblib"
    bad.write_bytes(b"not a pickle at all")
    with pytest.raises(ModelError, match="could not be loaded|not a usable"):
        load_trusted_model(bad)
    with pytest.raises(ModelError, match="not found"):
        load_trusted_model(tmp_path / "missing.joblib")
    path = _artifact(tmp_path)
    (tmp_path / "model.json").write_text(json.dumps({"model_name": "other"}))
    with pytest.raises(ModelError, match="model.json"):
        load_trusted_model(path)
    (tmp_path / "model.json").unlink()
    assert any("model.json not found" in w for w in load_trusted_model(path).warnings)


def test_missing_classifier_package_gives_an_actionable_message(tmp_path, monkeypatch):
    f = tmp_path / "model.joblib"
    f.write_bytes(b"x")
    monkeypatch.setitem(sys.modules, "emg_classifier", None)
    ok, msg = classifier_status()
    assert not ok and "not installed" in msg and "README" in msg
    with pytest.raises(ModelError, match="cannot be imported"):
        load_trusted_model(f)
