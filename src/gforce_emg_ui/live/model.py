"""Load and verify the trusted, frozen emg-classifier model for live use.

emg_classifier (and scikit-learn) are imported here, lazily, on the loader thread.
Model files are joblib/pickle: loading one runs code from the file, so only an
explicitly selected local file the user trusts is loaded, never one found
automatically. Missing or contradictory metadata is an error, never guessed.
"""

import hashlib
import importlib.util
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

TRUST_NOTE = ("Model files are joblib/pickle: loading one can run arbitrary code from the "
              "file. Load only a model you trained yourself and keep on this machine.")
INSTALL_HINT = "see README (Live Classifier) to install emg-classifier into this environment"
PRIMARY = "short_window_lda"
REJECTED = {
    "whole_hold_lda": "a whole-HOLD baseline (one prediction per complete HOLD, not "
                      "live-compatible)",
    "global_amplitude_lda": "the global-amplitude baseline",
    "majority_dummy": "the majority-class baseline",
}


class ModelError(Exception):
    """The selected file cannot be used for live classification (message is user-facing)."""


@dataclass(frozen=True)
class LoadedModel:
    clf: Any                        # emg_classifier.EffortClassifier (frozen)
    path: Path
    sha256: str                     # of model.joblib
    sidecar_sha256: Optional[str]   # of model.json next to it, if present
    name: str
    window_samples: int
    hop_samples: int
    window_units: str
    channel_names: Tuple[str, ...]
    classes: Tuple[Tuple[int, str, float], ...]
    feature: dict
    acquisition: dict
    training: dict
    cv_reference: dict
    hold_rule: dict
    created: str
    code_sha256: Optional[str]      # classifier sources the model was trained with
    installed_code_sha256: str      # installed emg_classifier sources
    trained_versions: dict
    installed_versions: dict
    limitations: Tuple[str, ...]
    warnings: Tuple[str, ...]       # need acknowledgement before Start

    @property
    def short_id(self) -> str:
        return f"{self.name} · sha256 {self.sha256[:12]}"

    def identity(self) -> dict:
        """JSON-able identity for run logs."""
        return {"path": str(self.path), "sha256": self.sha256,
                "sidecar_sha256": self.sidecar_sha256, "name": self.name,
                "window_samples": self.window_samples, "hop_samples": self.hop_samples,
                "window_units": self.window_units, "channel_names": list(self.channel_names),
                "classes": [list(c) for c in self.classes], "feature": self.feature,
                "created": self.created, "code_sha256": self.code_sha256,
                "installed_code_sha256": self.installed_code_sha256,
                "trained_versions": self.trained_versions,
                "installed_versions": self.installed_versions,
                "training": {k: self.training.get(k) for k in
                             ("n_trials", "n_windows", "per_class", "dataset_sha256")},
                "cv_reference": self.cv_reference, "warnings": list(self.warnings)}


def classifier_status() -> Tuple[bool, str]:
    """Is emg_classifier importable? Cheap: does not import it or scikit-learn."""
    try:
        found = importlib.util.find_spec("emg_classifier") is not None
    except (ImportError, ValueError):
        found = False
    if found:
        return True, ""
    return False, f"emg-classifier is not installed in {sys.executable}: {INSTALL_HINT}."


def package_sha256(pkg_dir: Path) -> str:
    """emg_classifier.provenance.code_hash() recipe applied to the installed package (that
    function hashes the source checkout, not site-packages). CRLF is read as LF: Git on
    Windows checks out / archives the same sources with CRLF line endings."""
    h = hashlib.sha256()
    for p in sorted(Path(pkg_dir).glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def _versions() -> dict:
    from importlib import metadata
    out = {}
    for name in ("emg-classifier", "scikit-learn", "numpy", "scipy", "joblib"):
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _need(meta: dict, key: str, kind, what: str):
    v = meta.get(key)
    if v is None or not isinstance(v, kind) or isinstance(v, bool) and kind is not bool:
        raise ModelError(f"model metadata `{key}` ({what}) is missing or invalid: {v!r}")
    return v


def load_trusted_model(path) -> LoadedModel:
    """Deserialize (trusted file only) and verify a short-window model. Raises ModelError."""
    try:
        import emg_classifier
        from emg_classifier import api, constants, features
    except ImportError as e:
        raise ModelError(f"emg-classifier cannot be imported in {sys.executable} ({e}); "
                         f"{INSTALL_HINT}.") from None
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ModelError(f"model file not found: {path}")
    sha = _sha256(path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            clf = api.load_model(path)
        except ValueError as e:
            raise ModelError(f"{path.name} is not a usable emg-classifier model: {e}") from None
        except Exception as e:                    # corrupt / truncated / foreign pickle
            raise ModelError(f"{path.name} could not be loaded: {type(e).__name__}: {e}") from None
    version_warnings = [str(w.message) for w in caught
                        if "version" in str(w.message).lower()
                        or type(w.message).__name__ == "InconsistentVersionWarning"]
    meta = clf.metadata
    installed = _versions()
    trained = meta.get("package_versions") or {}

    name = _need(meta, "model_name", str, "model type")
    if name in REJECTED:
        raise ModelError(f"{path.name} is {REJECTED[name]}; the Live Classifier needs the "
                         f"short-window model ({PRIMARY})")
    if name != PRIMARY:
        raise ModelError(f"unknown model type {name!r}; only {PRIMARY} is supported")
    if trained.get("scikit-learn") != installed["scikit-learn"] or version_warnings:
        raise ModelError(f"model trained with scikit-learn {trained.get('scikit-learn')}, this "
                         f"environment has {installed['scikit-learn']} "
                         f"({'; '.join(version_warnings) or 'version mismatch'}); install the "
                         f"pinned versions (constraints-live.txt)")

    N = _need(meta, "window_samples", int, "window length")
    H = _need(meta, "hop_samples", int, "hop")
    if N < 2 or H < 1:
        raise ModelError(f"window/hop must be positive sample counts, got {N}/{H}")
    units = _need(meta, "window_units", str, "window units")
    if "sample" not in units.lower():
        raise ModelError(f"window units {units!r} are not sample counts")
    if N != clf.window_samples:
        raise ModelError(f"window_samples {N} disagrees with the classifier ({clf.window_samples})")
    channels = tuple(_need(meta, "channel_names", list, "channel order"))
    feature = _need(meta, "feature", dict, "feature definition")
    if (feature.get("name"), feature.get("version")) != (features.FEATURE_NAME,
                                                         features.FEATURE_VERSION):
        raise ModelError(f"feature {feature} is not this classifier's "
                         f"{features.FEATURE_NAME} v{features.FEATURE_VERSION}")
    if len(meta.get("feature_order") or []) != len(channels):
        raise ModelError("feature_order does not have one feature per channel")

    classes = []
    for c in _need(meta, "classes", list, "class mapping"):
        cid, label, s = c.get("class_id"), c.get("label"), c.get("s")
        if constants.CLASS_ID.get(label) != cid or constants.S_VALUE.get(label) != s:
            raise ModelError(f"class mapping {c} contradicts the classifier's "
                             f"{constants.CLASS_ID} / {constants.S_VALUE}")
        classes.append((int(cid), str(label), float(s)))
    classes.sort()
    if [c[0] for c in classes] != list(constants.CLASS_IDS):
        raise ModelError(f"classes {classes} are not exactly {constants.CLASS_IDS}")

    steps = getattr(clf.pipeline, "steps", None) or []
    kinds = [type(est).__name__ for _, est in steps]
    if kinds != ["StandardScaler", "LinearDiscriminantAnalysis"]:
        raise ModelError(f"pipeline {kinds} is not StandardScaler -> LinearDiscriminantAnalysis")
    if getattr(clf.pipeline, "n_features_in_", None) != len(channels):
        raise ModelError(f"model expects {getattr(clf.pipeline, 'n_features_in_', None)} "
                         f"features, metadata lists {len(channels)} channels")
    if sorted(int(c) for c in clf.pipeline.classes_) != list(constants.CLASS_IDS):
        raise ModelError(f"model.classes_ {list(clf.pipeline.classes_)} != {constants.CLASS_IDS}")

    training = _need(meta, "training_set", dict, "training set")
    n_trials, n_windows = training.get("n_trials"), training.get("n_windows")
    per_trial = meta.get("training_windows_per_trial")
    seen = getattr(steps[0][1], "n_samples_seen_", None)
    if not all(isinstance(v, int) and v > 0 for v in (n_trials, n_windows, per_trial)):
        raise ModelError("training_set n_trials / n_windows or training_windows_per_trial missing")
    if n_trials * per_trial != n_windows:
        raise ModelError(f"{n_trials} trials x {per_trial} windows != {n_windows} training windows")
    if seen is None or int(seen) != n_windows:
        raise ModelError(f"the scaler was fitted on {seen} windows but the metadata lists "
                         f"{n_windows}: not the final model fitted on all training trials")

    acq = _need(meta, "acquisition", dict, "acquisition assumptions")
    keys, configs = acq.get("keys") or [], acq.get("configurations") or {}
    if not keys or any(not (configs.get(k) or {}).get("summary") for k in keys):
        raise ModelError("acquisition configurations missing: compatibility cannot be checked")

    model_warnings = []
    sidecar = path.with_name("model.json")
    sidecar_sha = None
    if sidecar.is_file():
        sidecar_sha = _sha256(sidecar)
        try:
            same = json.loads(sidecar.read_text()) == json.loads(json.dumps(meta))
        except ValueError:
            same = False
        if not same:
            raise ModelError("model.json next to the model differs from the metadata inside "
                             f"{path.name}: resolve which file is the selected export")
    else:
        model_warnings.append("model.json not found next to the model (embedded metadata used)")
    pkg_dir = Path(emg_classifier.__file__).resolve().parent
    installed_code = package_sha256(pkg_dir)
    code = meta.get("code_sha256")
    if code != installed_code:
        model_warnings.append(f"installed emg-classifier sources ({installed_code[:12]}) differ "
                              f"from the sources the model was trained with "
                              f"({str(code)[:12]})")
    for pkg in ("numpy", "scipy", "joblib"):
        if trained.get(pkg) and trained.get(pkg) != installed.get(pkg):
            model_warnings.append(f"{pkg} {installed.get(pkg)} here, model trained with "
                                  f"{trained.get(pkg)}")
    return LoadedModel(
        clf=clf, path=path, sha256=sha, sidecar_sha256=sidecar_sha, name=name,
        window_samples=N, hop_samples=H, window_units=units, channel_names=channels,
        classes=tuple(classes), feature=dict(feature), acquisition=acq, training=training,
        cv_reference=dict(meta.get("cv_reference") or {}), hold_rule=dict(meta.get("hold_rule")
                                                                          or {}),
        created=str(meta.get("created")), code_sha256=code, installed_code_sha256=installed_code,
        trained_versions=dict(trained), installed_versions=installed,
        limitations=tuple(meta.get("limitations") or ()), warnings=tuple(model_warnings))
