"""Trial protocol for stiffness-command data collection (Qt-free).

One instructed effort level per trial. Phases (default durations):
  PREPARE 5 s -> SETTLE 3 s -> HOLD 5 s -> REST 15 s   (28 s recorded)
Only HOLD is a training-candidate interval. Phase and target are separate:
a REST interval is never "low", whatever the trial's target. Each trial also has
one motion condition; in moving trials the movement starts in SETTLE and continues
through HOLD, so HOLD means "maintain the effort", not "stay still".

Labels are INSTRUCTED EFFORT / INTENDED ROBOT GAIN levels. They are not measured
physical human stiffness, force, %MVC or verified antagonist co-contraction.

The state machine uses an injected monotonic clock and deadlines (no sleeps,
no decrement counters). update() performs at most one transition per call, and
the next phase starts when update() actually ran, so a delayed GUI tick makes
the current phase longer instead of backdating an unshown cue or skipping a
phase. Overruns are recorded; a large overrun before or during HOLD aborts the
trial instead of producing a HOLD whose timing is unusable.
"""

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, List, Optional

SCHEMA_VERSION = 2                      # 2: motion_condition + movement instruction
PROTOCOL_VERSION = "stiffness-phase0/1.3"     # 1.3: contact never blocks or flags
LABEL_SOURCE = "instructed_effort"
LABEL_SEMANTICS = ("Instructed effort level / intended robot gain command. NOT measured physical "
                   "human stiffness, force, %MVC, EMG amplitude or verified co-contraction.")

# Explicit, stable label mapping (target_label -> class id and command value s)
TARGETS = {
    "low": {"class_id": 0, "s": 0.0, "display": "LOW"},
    "medium": {"class_id": 1, "s": 0.5, "display": "MEDIUM"},
    "high": {"class_id": 2, "s": 1.0, "display": "HIGH"},
}
TARGET_ORDER = ("low", "medium", "high")

# Shown to the participant (SETTLE) and saved verbatim with every trial
TARGET_INSTRUCTIONS = {
    "low": "Keep the selected grasp/fist with the minimum comfortable effort needed. "
           "Do not open the hand.",
    "medium": "Use a repeatable moderate effort, clearly between low and high.",
    "high": "Use a firm, comfortable, repeatable effort (not maximal). "
            "Stop or rest longer if you feel discomfort or fatigue.",
}
# Shown to the operator only
OPERATOR_NOTES = {
    "low": "s=0 is not an open hand: the participant keeps the selected grasp/fist.",
    "medium": "Medium is NOT a measured 50% of force, EMG, MVC or human stiffness.",
    "high": "Never ask for maximal or 'as hard as possible' squeezes.",
}

PREPARE, SETTLE, HOLD, REST = "PREPARE", "SETTLE", "HOLD", "REST"
PHASES = (PREPARE, SETTLE, HOLD, REST)
IDLE, DONE, ABORTED = "IDLE", "DONE", "ABORTED"
TRAINING_CANDIDATE_PHASE = HOLD

# Motion conditions (stable IDs). Movement is instructed, not measured.
MOTION_INSTRUCTION_VERSION = "motion/1"
STATIONARY = "stationary"
MOTIONS = {
    STATIONARY: {"display": "Stationary", "movement": None},
    "wrist_flex_ext": {"display": "Wrist flexion/extension",
                       "movement": "slowly bend the wrist up and down through a comfortable range"},
    "forearm_rotation": {"display": "Forearm rotation",
                         "movement": "slowly turn the forearm palm-up and palm-down through a "
                                     "comfortable range"},
    "reach_sweep": {"display": "Reach/sweep",
                    "movement": "slowly reach forward and sweep the arm side to side through a "
                                "comfortable range"},
}
MOTION_ORDER = (STATIONARY, "wrist_flex_ext", "forearm_rotation", "reach_sweep")
MOVEMENT_PHASES = (SETTLE, HOLD)

# Marker labels written to markers.csv (stable, machine-readable)
MARKERS = ("PREPARE_START", "SETTLE_START", "HOLD_START", "HOLD_END", "REST_START",
           "REST_END", "REST_EXTENDED", "ABORT", "TRIAL_END")


def target_info(label: str) -> dict:
    t = TARGETS[label]
    return {"target_label": label, "target_class_id": t["class_id"], "target_s": t["s"],
            "label_source": LABEL_SOURCE}


def motion_info(motion: str) -> dict:
    m = MOTIONS[motion]
    return {"motion_condition": motion, "display": m["display"],
            "instruction_version": MOTION_INSTRUCTION_VERSION,
            "movement_instruction": m["movement"],
            "movement_phases": [] if m["movement"] is None else list(MOVEMENT_PHASES)}


def phase_instruction(phase: str, target: str, motion: str = STATIONARY) -> str:
    level = TARGETS[target]["display"]
    move = MOTIONS[motion]["movement"]
    if phase == PREPARE:
        text = ("Position your hand/object. Keep the intended hand shape with minimal "
                "comfortable tension.")
        if move:
            text += f" Movement this trial: {MOTIONS[motion]['display'].lower()} (starts in SETTLE)."
        return text
    if phase == SETTLE:
        if move:
            return (f"Smoothly establish {level} effort and begin the movement: {move}. "
                    f"{TARGET_INSTRUCTIONS[target]}")
        return f"Smoothly establish {level} effort. {TARGET_INSTRUCTIONS[target]}"
    if phase == HOLD:
        if move:
            return (f"Maintain {level} effort while you continue the movement: {move}. "
                    "Keep the same grasp/hand shape.")
        return f"Maintain {level} effort. Keep your hand and wrist position steady."
    if phase == REST:
        prefix = "Stop the movement, release" if move else "Release"
        return f"{prefix} the effort and rest. Set down/support the object as needed."
    if phase == ABORTED:
        return "Trial stopped. " + ("Stop moving, release" if move else "Release") + \
            " the effort and rest."
    return ""


def participant_instructions(target: str, motion: str = STATIONARY) -> dict:
    """Exact text shown in each phase, saved with the trial."""
    return {p: phase_instruction(p, target, motion) for p in PHASES + (ABORTED,)}


@dataclass
class ProtocolSettings:
    prepare_s: float = 5.0
    settle_s: float = 3.0
    hold_s: float = 5.0
    rest_s: float = 15.0
    rest_extension_s: float = 15.0      # per "Extend rest" click
    readiness_s: float = 3.0            # readiness must hold this long before Start
    emg_freshness_s: float = 0.5        # max age of the newest received EMG batch (software
                                        # setting; verify against the band's delivery bursts)
    overrun_flag_s: float = 0.2         # phase overrun recorded as a timing flag
    overrun_abort_s: float = 1.0        # overrun in PREPARE/SETTLE/HOLD aborts the trial
    hold_quality_tail_s: float = 0.5    # batches received this long after HOLD_END still
                                        # count for HOLD quality (they carry HOLD samples)
    audio_cues: bool = False            # optional short beep on phase changes

    MAX_S = 3600.0

    def durations(self) -> dict:
        return {PREPARE: self.prepare_s, SETTLE: self.settle_s, HOLD: self.hold_s,
                REST: self.rest_s}

    @property
    def total_s(self) -> float:
        return sum(self.durations().values())

    def validate(self) -> List[str]:
        problems = []

        def check(name, value, allow_zero=False):
            ok = isinstance(value, (int, float)) and not isinstance(value, bool) \
                and math.isfinite(value) and (value >= 0 if allow_zero else value > 0) \
                and value <= self.MAX_S
            if not ok:
                problems.append(f"{name} must be a finite {'non-negative' if allow_zero else 'positive'}"
                                f" number of seconds (≤ {self.MAX_S:g}), got {value!r}")
        for name in ("prepare_s", "settle_s", "hold_s", "rest_s", "rest_extension_s",
                     "emg_freshness_s", "overrun_flag_s", "overrun_abort_s"):
            check(name, getattr(self, name))
        for name in ("readiness_s", "hold_quality_tail_s"):
            check(name, getattr(self, name), allow_zero=True)
        if not problems and self.overrun_abort_s <= self.overrun_flag_s:
            problems.append("overrun_abort_s must be larger than overrun_flag_s")
        return problems

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ProtocolEvent:
    name: str                 # marker label, e.g. "HOLD_START"
    host_mono: float          # same monotonic timebase as the recorder's host_mono
    host_time: float          # wall clock, for cross-referencing only
    phase: str                # machine phase after the event
    detail: dict = field(default_factory=dict)


class TrialStateMachine:
    """PREPARE -> SETTLE -> HOLD -> REST -> DONE, or ABORTED from any active phase."""

    def __init__(self, settings: ProtocolSettings, clock: Callable[[], float] = time.monotonic,
                 wall: Callable[[], float] = time.time):
        problems = settings.validate()
        if problems:
            raise ValueError("; ".join(problems))
        self.settings = ProtocolSettings(**settings.to_dict())     # frozen copy
        self._clock = clock
        self._wall = wall
        self._planned = self.settings.durations()
        self.phase = IDLE
        self.events: List[ProtocolEvent] = []
        self.intervals: List[dict] = []
        self.flags: List[dict] = []
        self.abort_reason: Optional[str] = None
        self.rest_extensions = 0
        self.start_mono: Optional[float] = None
        self.end_mono: Optional[float] = None
        self._deadline: Optional[float] = None
        self._last_tick: Optional[float] = None
        self._stop_pending = False

    # -- state ------------------------------------------------------------------
    @property
    def active(self) -> bool:
        return self.phase in PHASES

    @property
    def finished(self) -> bool:
        return self.phase in (DONE, ABORTED)

    @property
    def current(self) -> Optional[dict]:
        return self.intervals[-1] if self.active else None

    def phase_remaining(self, now: Optional[float] = None) -> float:
        if not self.active:
            return 0.0
        now = self._clock() if now is None else now
        return max(0.0, self._deadline - now)

    def elapsed(self, now: Optional[float] = None) -> float:
        if self.start_mono is None:
            return 0.0
        end = self.end_mono if self.end_mono is not None else (self._clock() if now is None else now)
        return end - self.start_mono

    def total_remaining(self, now: Optional[float] = None) -> float:
        """Remaining time of the current phase plus the planned later phases."""
        if not self.active:
            return 0.0
        later = PHASES[PHASES.index(self.phase) + 1:]
        return self.phase_remaining(now) + sum(self._planned[p] for p in later)

    def consume_stop_request(self) -> bool:
        """True exactly once, after the protocol ended (DONE or ABORTED)."""
        if self._stop_pending:
            self._stop_pending = False
            return True
        return False

    # -- transitions --------------------------------------------------------------
    def _event(self, name, now, wall, detail=None) -> ProtocolEvent:
        ev = ProtocolEvent(name, now, wall, self.phase, dict(detail or {}))
        self.events.append(ev)
        return ev

    def _enter(self, phase, now, wall):
        self.phase = phase
        self._deadline = now + self._planned[phase]
        self.intervals.append({"phase": phase, "start_host_mono": now, "start_host_time": wall,
                               "planned_s": self._planned[phase], "end_host_mono": None,
                               "end_host_time": None, "actual_s": None, "overrun_s": None,
                               "max_tick_gap_s": 0.0, "completed": False})

    def _close(self, now, wall, completed: bool, overrun: Optional[float]):
        iv = self.intervals[-1]
        iv.update(end_host_mono=now, end_host_time=wall, actual_s=now - iv["start_host_mono"],
                  overrun_s=overrun, completed=completed)

    def start(self) -> List[ProtocolEvent]:
        if self.phase != IDLE:
            raise RuntimeError(f"trial already started (phase {self.phase})")
        now, wall = self._clock(), self._wall()
        self.start_mono = now
        self._last_tick = now
        self._enter(PREPARE, now, wall)
        return [self._event("PREPARE_START", now, wall)]

    def update(self) -> List[ProtocolEvent]:
        """Call from a timer; performs at most one phase transition."""
        if not self.active:
            return []
        now = self._clock()
        iv = self.intervals[-1]
        gap = now - self._last_tick
        self._last_tick = now
        if gap > iv["max_tick_gap_s"]:
            iv["max_tick_gap_s"] = gap
        if gap > self.settings.overrun_flag_s:
            self.flags.append({"type": "tick_gap", "phase": self.phase, "gap_s": gap,
                               "host_mono": now})
        if now < self._deadline:
            return []
        overrun = now - self._deadline
        phase = self.phase
        if phase != REST and overrun > self.settings.overrun_abort_s:
            return self.abort("timing_overrun", {"phase": phase, "overrun_s": overrun})
        if overrun > self.settings.overrun_flag_s:
            self.flags.append({"type": "overrun", "phase": phase, "overrun_s": overrun,
                               "host_mono": now})
        wall = self._wall()
        self._close(now, wall, True, overrun)
        if phase == PREPARE:
            self._enter(SETTLE, now, wall)
            return [self._event("SETTLE_START", now, wall)]
        if phase == SETTLE:
            self._enter(HOLD, now, wall)
            return [self._event("HOLD_START", now, wall)]
        if phase == HOLD:
            end = self._event("HOLD_END", now, wall)
            self._enter(REST, now, wall)
            end.phase = REST
            return [end, self._event("REST_START", now, wall)]
        # REST finished: the protocol is complete
        self.phase = DONE
        self.end_mono = now
        self._stop_pending = True
        return [self._event("REST_END", now, wall), self._event("TRIAL_END", now, wall)]

    def extend_rest(self) -> List[ProtocolEvent]:
        if self.phase != REST:
            return []
        now, wall = self._clock(), self._wall()
        ext = self.settings.rest_extension_s
        self._deadline += ext
        self.rest_extensions += 1
        self.intervals[-1]["planned_s"] += ext
        return [self._event("REST_EXTENDED", now, wall,
                            {"extension_s": ext, "count": self.rest_extensions})]

    def abort(self, reason: str, detail: Optional[dict] = None) -> List[ProtocolEvent]:
        if not self.active:
            return []
        now, wall = self._clock(), self._wall()
        phase = self.phase
        self._close(now, wall, False, None)
        self.abort_reason = reason
        self.phase = ABORTED
        self.end_mono = now
        self._stop_pending = True
        info = {"reason": reason, "during": phase, **(detail or {})}
        return [self._event("ABORT", now, wall, info), self._event("TRIAL_END", now, wall)]
