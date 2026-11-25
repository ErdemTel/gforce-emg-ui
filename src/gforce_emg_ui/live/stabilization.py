"""Output stabilization after inference (Qt-free): time-based class confirmation.

  raw prediction (unchanged) -> Stabilizer.observe() -> accepted class, or unavailable

A raw winner that differs from the accepted class becomes a pending candidate. It is
accepted once that same candidate has won every processed observation, without a break,
for at least the confirmation time: upward / downward by s order (0 < 0.5 < 1), or the
startup time while no class is accepted. Direct low <-> high changes are allowed; nothing
bypasses confirmation (0 ms = accept on the first qualifying observation).

Time is the host receipt time of each window's newest sample (LiveResult
.newest_receipt_mono): host-observed persistence, not acquisition time or physiological
latency. Observations sharing a receipt time add 0 ms; inference or GUI speed adds
nothing. The confirmation is extra delay on top of windowing, transport and inference.
There are no timers: only observe() can accept a class. Freshness (now - newest evidence
<= the engine's stale_s) is checked when the output is read, as for the raw output.

Optional probability gate on classifier scores (not calibrated certainty): the candidate
needs p >= min_probability and, if a class is accepted, p - p(accepted) >= min_margin.
"Support" for the accepted class = a fresh valid observation whose raw winner is that
class (the gate applies only to candidates). If fresh observations keep arriving but none
supports the accepted class for longer than max_unsupported_ms, the output becomes
unavailable (unsupported hold expired), which is separate from stale input.

Disabled = pass-through: accepted = raw winner of every valid observation.
The caller (LiveEngine) serializes all calls.
"""

import collections
import json
import math
import numbers
from dataclasses import asdict, dataclass, fields
from typing import List, NamedTuple, Optional

LABELS = ("low", "medium", "high")       # known labels; a model may use a subset (>= 2)
S_VALUE = {"low": 0.0, "medium": 0.5, "high": 1.0}
ALGORITHM = "persistence_confirmation"
ALGORITHM_VERSION = 1
SETTINGS_SCHEMA = 1
TIMING_NOTE = ("confirmation time = host receipt time of the newest window sample "
               "(host-observed persistence; not acquisition time, not physiological latency; "
               "adds to window, transport and inference delay)")
EPS_MS = 1e-6                       # float rounding of receipt-time differences

OFF, WAITING, CONFIRMING, HOLDING, PENDING, GATE_NOT_MET, UNSUPPORTED = (
    "OFF", "WAITING", "CONFIRMING", "HOLDING", "PENDING", "GATE_NOT_MET", "UNSUPPORTED")

# field -> (min, max); max None = the engine's stale_s
LIMITS = {"up_ms": (0, 1000), "down_ms": (0, 1000), "startup_ms": (0, 1000),
          "min_probability": (0.0, 1.0), "min_margin": (0.0, 1.0),
          "max_gap_ms": (1, None), "max_unsupported_ms": (100, 10000)}


class SettingsError(ValueError):
    pass


@dataclass(frozen=True)
class StabilizerSettings:
    """Initial engineering values, not validated optima."""
    enabled: bool = False
    up_ms: int = 100                # confirmation for a higher s
    down_ms: int = 100              # confirmation for a lower s
    startup_ms: int = 100           # confirmation while no class is accepted
    gate: bool = False              # probability gate (advanced)
    min_probability: float = 0.60
    min_margin: float = 0.10
    max_gap_ms: int = 250           # larger receipt-time gap between observations: restart
    max_unsupported_ms: int = 1000  # accepted class unsupported longer: unavailable

    def errors(self, stale_ms: float) -> List[str]:
        out = []
        for f in fields(self):
            v = getattr(self, f.name)
            if f.type in (bool, "bool"):
                if not isinstance(v, bool):
                    out.append(f"{f.name} must be true/false, got {v!r}")
                continue
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                out.append(f"{f.name} must be a finite number, got {v!r}")
                continue
            if f.type in (int, "int") and v != int(v):
                out.append(f"{f.name} must be whole milliseconds, got {v!r}")
            lo, hi = LIMITS[f.name]
            hi = stale_ms if hi is None else hi
            if not lo <= v <= hi:
                out.append(f"{f.name} {v:g} outside {lo:g}–{hi:g}")
        if not out:
            need = max(self.up_ms, self.down_ms) + self.max_gap_ms
            if self.max_unsupported_ms < need:
                out.append(f"max_unsupported_ms {self.max_unsupported_ms} must be >= longer "
                           f"confirmation + max_gap_ms ({need}) so an ordinary confirmed "
                           f"change cannot expire first")
        return out

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d, stale_ms: float) -> "StabilizerSettings":
        """Complete, validated settings or SettingsError (never partly applied)."""
        if not isinstance(d, dict):
            raise SettingsError(f"settings must be a mapping, got {type(d).__name__}")
        names = {f.name for f in fields(cls)}
        missing, extra = names - set(d), set(d) - names
        if missing or extra:
            raise SettingsError(f"settings keys: missing {sorted(missing)}, "
                                f"unknown {sorted(extra)}")
        kw = {}
        for f in fields(cls):
            v = d[f.name]
            if f.type in (int, "int") and isinstance(v, float) and v.is_integer():
                v = int(v)
            kw[f.name] = v
        s = cls(**kw)
        errs = s.errors(stale_ms)
        if errs:
            raise SettingsError("; ".join(errs))
        return s


class Observation(NamedTuple):
    pred_id: int
    run_id: Optional[str]
    generation: int
    stream_id: int
    segment: int
    window_end_index: int
    hop_samples: int
    evidence_mono: float            # newest_receipt_mono of the window
    result_mono: float
    valid: bool
    label: Optional[str]
    probabilities: Optional[dict]


def observation(r, hop_samples: int) -> Observation:
    """From a LiveResult (duck-typed); the Prediction is read, never changed."""
    p = r.prediction
    return Observation(r.pred_id, r.run_id, r.generation, r.stream_id, r.segment,
                       r.window_end_index, hop_samples, r.newest_receipt_mono, r.result_mono,
                       bool(p.valid), p.label, p.probabilities)


@dataclass(frozen=True)
class StabilizerState:
    enabled: bool
    revision: int
    settings: StabilizerSettings
    status: str
    reason: str                     # last reset / cancel / expiry reason
    accepted: Optional[str]
    accepted_pred_id: Optional[int]           # the observation that established it
    accepted_start_mono: Optional[float]      # evidence: its confirmation began
    accepted_evidence_mono: Optional[float]   # evidence: confirmation completed
    accepted_decision_mono: Optional[float]   # host clock when it was accepted
    pending: Optional[str]
    pending_start_mono: Optional[float]
    pending_elapsed_ms: Optional[float]       # processed evidence only
    pending_required_ms: Optional[float]
    raw_label: Optional[str]                  # winner of the last processed observation
    last_pred_id: Optional[int]
    last_evidence_mono: Optional[float]
    last_support_mono: Optional[float]
    counters: dict

    @property
    def s(self) -> Optional[float]:
        return None if self.accepted is None else S_VALUE[self.accepted]


def describe(st: StabilizerState) -> str:
    """One-line status for the UI, from processed evidence only."""
    if st.status == OFF:
        return "Stabilization OFF"
    pend = ""
    if st.pending is not None:
        pend = (f"{st.pending.upper()} pending: {st.pending_elapsed_ms:.0f} / "
                f"{st.pending_required_ms:.0f} ms")
    if st.status == UNSUPPORTED:
        return "Output unavailable: unsupported hold expired" + (f"; {pend}" if pend else "")
    if st.accepted is None:
        if st.status == CONFIRMING:
            return f"Confirming initial class: {pend}"
        if st.status == GATE_NOT_MET:
            return f"Probability gate not met ({(st.raw_label or '').upper()}); no class accepted"
        return "Waiting for predictions" + (f" ({st.reason})" if st.reason else "")
    hold = f"Holding {st.accepted.upper()}"
    if st.status == PENDING:
        return f"{hold}; {pend}"
    if st.status == GATE_NOT_MET:
        return f"{hold}; probability gate not met ({(st.raw_label or '').upper()})"
    return hold


STABILIZATION_FIELDS = [
    "kind", "t_mono", "run_id", "generation", "stream_id", "segment", "pred_id",
    "window_end_index", "newest_receipt_mono", "result_mono", "revision", "enabled",
    "raw_valid", "raw_label", "p_low", "p_medium", "p_high", "output_valid", "accepted_label",
    "accepted_s", "accepted_pred_id", "accepted_start_mono", "accepted_evidence_mono",
    "accepted_decision_mono", "pending_label", "pending_start_mono", "pending_elapsed_ms",
    "pending_required_ms", "last_support_mono", "status", "transition", "reason", "settings"]


class Stabilizer:
    def __init__(self, settings: Optional[StabilizerSettings] = None):
        self.settings = settings or StabilizerSettings()
        self.revision = 0
        self.counters = collections.Counter()
        self._last_pred: Optional[int] = None     # duplicates are refused across resets
        self._boundary = None                     # (mono, pred_id): older evidence unused
        self._clear("")

    # -- control ------------------------------------------------------------------
    def configure(self, settings: StabilizerSettings, now: float, run_id=None,
                  generation=None) -> dict:
        """New revision; state cleared; evidence received before `now` is not used."""
        self.settings = settings
        self.revision += 1
        self.counters["settings_applied"] += 1
        self._boundary = (now, self._last_pred)
        self._clear("settings_applied")
        return self._row("settings", now, None, "settings_applied", run_id, generation)

    def reset(self, reason: str, now: float, run_id=None, generation=None,
              boundary: bool = False, new_run: bool = False) -> dict:
        """Clear the state (show unavailable). boundary: also refuse evidence older than now.
        Other resets keep an earlier boundary (results queued before Apply stay unused)."""
        if new_run:
            self.counters.clear()
            self._boundary = None
        self.counters[f"reset:{reason}"] += 1
        if boundary:
            self._boundary = (now, self._last_pred)
        self._clear(reason)
        return self._row("reset", now, None, reason, run_id, generation)

    def settings_row(self, now: float, run_id=None, generation=None) -> dict:
        """The active settings, e.g. at run start."""
        return self._row("settings", now, None, "active", run_id, generation)

    def break_evidence(self, reason: str, now: float, run_id=None, generation=None) -> dict:
        """A window could not be used (skipped / expired): the next one is not consecutive."""
        self._cancel(reason)
        self._gap_ref = self._last_end = None
        return self._row("break", now, None, reason, run_id, generation)

    # -- observations -------------------------------------------------------------
    def observe(self, ob: Observation, now: float) -> dict:
        """Process one published, fresh prediction, in order. Returns its log row."""
        if self._last_pred is not None and ob.pred_id <= self._last_pred:
            self.counters["ignored_duplicate"] += 1
            return self._row("ignored", now, ob, "duplicate_or_out_of_order")
        b = self._boundary
        if b is not None and (ob.evidence_mono < b[0] or
                              (b[1] is not None and ob.pred_id <= b[1])):
            self._last_pred = ob.pred_id
            self.counters["ignored_before_boundary"] += 1
            return self._row("ignored", now, ob, "evidence_before_revision_boundary")
        self._last_pred = ob.pred_id
        ctx = (ob.run_id, ob.generation, ob.stream_id, ob.segment)
        if self._ctx is not None and ctx != self._ctx:
            self._clear("context_change")
            self.counters["reset:context_change"] += 1
        self._ctx = ctx
        probs = ob.probabilities or {}
        if not (ob.valid and ob.label in probs and len(probs) >= 2 and all(
                l in S_VALUE and isinstance(v, numbers.Real) and not isinstance(v, bool)
                and math.isfinite(v) for l, v in probs.items())):
            self._clear("invalid_prediction")
            self.counters["reset:invalid_prediction"] += 1
            return self._row("observe", now, ob, "invalid_prediction")
        s, t, label = self.settings, ob.evidence_mono, ob.label
        reason = ""
        if self._gap_ref is not None:
            dt_ms = (t - self._gap_ref) * 1000
            if dt_ms < 0:
                reason = "evidence_time_backward"
            elif dt_ms > s.max_gap_ms + EPS_MS:
                reason = "evidence_gap"
            elif ob.window_end_index != self._last_end + ob.hop_samples:
                reason = "window_discontinuity"
            if reason:
                self._cancel(reason)
        self._gap_ref, self._last_end = t, ob.window_end_index
        if self._raw is not None and label != self._raw:
            self.counters["raw_changes"] += 1
        self._raw, self._last_t = label, t
        self.counters["observations"] += 1
        self._gate_failed = False
        self._cancelled = ""
        transition = ""
        A = self._accepted
        if not s.enabled:                         # pass-through: raw winner, no gate
            if label != A:
                transition = self._accept(ob, label, t, now, t)
            self._support = t
        elif label == A:
            self._cancel("returned_to_accepted")
            self._support = t
        elif s.gate and not self._gate_ok(probs, label, A):
            self._cancel("gate_not_met")
            self._gate_failed = True
            self.counters["gate_failures"] += 1
            reason = reason or "gate_not_met"
            self._expire(t)
        else:
            if self._pending != label:
                self._cancel("candidate_changed")
                self._pending, self._pending_start = label, t
                self.counters["candidates_started"] += 1
            if self._elapsed_ms(t) + EPS_MS >= self._required_ms(label, A):
                transition = self._accept(ob, label, t, now, self._pending_start)
            else:
                self._expire(t)
        if self._expired_now:
            reason = "unsupported_hold_expired"
            self._expired_now = False
        if transition:
            reason = reason or ("confirmed" if s.enabled else "pass_through")
        return self._row("observe", now, ob, reason or self._cancelled, transition=transition)

    # -- state --------------------------------------------------------------------
    def state(self) -> StabilizerState:
        s = self.settings
        if not s.enabled:
            status = OFF
        elif self._accepted is None:
            status = (UNSUPPORTED if self._expired else CONFIRMING if self._pending
                      else GATE_NOT_MET if self._gate_failed else WAITING)
        else:
            status = (PENDING if self._pending else GATE_NOT_MET if self._gate_failed
                      else HOLDING)
        pend = self._pending
        return StabilizerState(
            s.enabled, self.revision, s, status, self._reason, self._accepted,
            self._acc_pred, self._acc_start, self._acc_evidence, self._acc_decision, pend,
            self._pending_start if pend else None,
            self._elapsed_ms(self._last_t) if pend else None,
            self._required_ms(pend, self._accepted) if pend else None,
            self._raw, self._last_pred, self._last_t, self._support, dict(self.counters))

    # -- internals ----------------------------------------------------------------
    def _clear(self, reason: str):
        self._accepted = self._acc_pred = None
        self._acc_start = self._acc_evidence = self._acc_decision = None
        self._pending = self._pending_start = None
        self._support = self._last_t = self._gap_ref = self._last_end = None
        self._raw = self._ctx = None
        self._gate_failed = self._expired = self._expired_now = False
        self._cancelled = ""
        self._reason = reason

    def _cancel(self, reason: str):
        if self._pending is not None:
            self.counters["cancelled_candidates"] += 1
            self.counters[f"cancelled:{reason}"] += 1
            self._pending = self._pending_start = None
            self._reason = self._cancelled = reason

    def _elapsed_ms(self, t: Optional[float]) -> float:
        return 0.0 if t is None or self._pending_start is None else \
            max(0.0, (t - self._pending_start) * 1000)

    def _required_ms(self, label: str, accepted: Optional[str]) -> float:
        s = self.settings
        if accepted is None:
            return float(s.startup_ms)
        return float(s.up_ms if S_VALUE[label] > S_VALUE[accepted] else s.down_ms)

    def _gate_ok(self, probs: dict, label: str, accepted: Optional[str]) -> bool:
        s = self.settings
        p = probs[label]
        if p < s.min_probability:
            return False
        return accepted is None or p - probs[accepted] >= s.min_margin

    def _accept(self, ob: Observation, label: str, t: float, now: float, start: float) -> str:
        prev = self._accepted
        self._accepted, self._acc_pred = label, ob.pred_id
        self._acc_start, self._acc_evidence, self._acc_decision = start, t, now
        self._pending = self._pending_start = None
        self._support = t
        self._expired = False
        self._reason = ""
        self.counters["initial_acceptances" if prev is None else "accepted_changes"] += 1
        return f"{prev or 'none'}->{label}"

    def _expire(self, t: float):
        A, s = self._accepted, self.settings
        if A is not None and (t - self._support) * 1000 > s.max_unsupported_ms + EPS_MS:
            self._accepted = self._acc_pred = None
            self._acc_start = self._acc_evidence = self._acc_decision = None
            self._expired = self._expired_now = True
            self._reason = "unsupported_hold_expired"
            self.counters["unsupported_expiries"] += 1

    def _row(self, kind: str, now: float, ob: Optional[Observation], reason: str,
             run_id=None, generation=None, transition: str = "") -> dict:
        st = self.state()
        probs = (ob.probabilities or {}) if ob is not None else {}
        row = {"kind": kind, "t_mono": now, "revision": self.revision,
               "enabled": int(st.enabled), "output_valid": int(st.accepted is not None),
               "accepted_label": st.accepted or "", "accepted_s": st.s,
               "accepted_pred_id": st.accepted_pred_id,
               "accepted_start_mono": st.accepted_start_mono,
               "accepted_evidence_mono": st.accepted_evidence_mono,
               "accepted_decision_mono": st.accepted_decision_mono,
               "pending_label": st.pending or "", "pending_start_mono": st.pending_start_mono,
               "pending_elapsed_ms": st.pending_elapsed_ms,
               "pending_required_ms": st.pending_required_ms,
               "last_support_mono": st.last_support_mono, "status": st.status,
               "transition": transition, "reason": reason or "",
               "run_id": run_id, "generation": generation}
        if ob is not None:
            row.update(run_id=ob.run_id, generation=ob.generation, stream_id=ob.stream_id,
                       segment=ob.segment, pred_id=ob.pred_id,
                       window_end_index=ob.window_end_index,
                       newest_receipt_mono=ob.evidence_mono, result_mono=ob.result_mono,
                       raw_valid=int(ob.valid), raw_label=ob.label or "",
                       p_low=probs.get("low"), p_medium=probs.get("medium"),
                       p_high=probs.get("high"))
        if kind == "settings":
            row["settings"] = json.dumps(self.settings.to_dict(), sort_keys=True)
        return row
