"""Runs one stiffness-collection attempt at a time on the shared Recorder (Qt-free).

controller.state:
  idle -> running       protocol phases; this controller owns the Recorder session
       -> finalizing    stop requested exactly once; wait for the writer, then trial.json
       -> review        outcome known; operator Accepts / Rejects / Repeats
       -> saving_review -> idle

Raw data is recorded by the existing Recorder (emg/imu/contact/markers.csv and
session.json, unchanged formats). Protocol events go to markers.csv through the
existing marker path, with the same host timestamps as in trial.json. trial.json
is written atomically on the I/O thread: at start (identity, status
"recording"), after the recorder finished ("finalized"), and after review
("reviewed"). An attempt is shown as saved only when the recorder reported
SAVED and trial.json was written. Progress is recomputed from trial.json files.
Only accepted attempts are kept: Reject / Repeat (and closing the app before
review) delete the attempt folder, aborted or not.

Plan mode: the next planned trial is a random proposal (plan_cursor). The
operator may edit its object, target or movement for that one trial
(edit_selection); `pending` then resolves to an incomplete planned trial with
exactly those values, or holds the reason there is none (Start blocked).
clear_edits returns to the proposal. Start re-resolves and freezes a copy of
the spec and its selection provenance for the attempt. Surface press is
stationary only: choosing it as the object also sets the movement to stationary.
Open hand keeps the requested movement.

extend_plan appends one task condition's trials (30 surface press or 35 open
hand) to the selected plan file on the I/O thread (idle only; Start, plan and
participant changes wait until it finished).
"""

import copy
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from .. import __version__
from ..config import EMG_CHANNELS, IMU_LABELS
from ..recorder import ERROR, SAVED
from ..sources import RECONNECTING, STREAMING
from .plan import (EMPTY_HAND, EXTENDABLE, EXTENSIONS, INCOMPATIBLE, MANUAL_SELECTION_RULE,
                   MAX_TARGET_RUN, OBJECTS_FILE, SELECTION_FIELDS, SELECTION_RULE,
                   TASK_CONDITIONS, TASKS, Progress, TrialPlan, attempt_category,
                   compute_progress, eligible_trials, extend_plan_file, extension_status,
                   load_objects, merge_objects, objects_doc, participant_root, pick_trial,
                   resolve_request, run_banned_target, scan_participant, trial_values)
from .protocol import (DONE, HOLD, LABEL_SEMANTICS, MOTIONS, PHASES, PROTOCOL_VERSION,
                       SCHEMA_VERSION, STATIONARY, SURFACE_PRESS, TARGETS,
                       TRAINING_CANDIDATE_PHASE, ProtocolEvent, ProtocolSettings,
                       TrialStateMachine, motion_info, target_info, task_instructions, trial_type)
from .readiness import (Readiness, ReadinessMonitor, StreamTracker, contact_events,
                        summarize_window)
from .storage import (IoWorker, atomic_write_text, attempt_name, discard_attempt, dumps,
                      ensure_within, git_info, valid_id, valid_key)

IDLE, RUNNING, FINALIZING, REVIEW, SAVING_REVIEW = (
    "idle", "running", "finalizing", "review", "saving_review")
ROLES = ("development", "held_out_test", "practice")
HAND_SIDES = ("right", "left")
REVIEW_REASONS = {                      # IDs are stable (saved in trial.json)
    "wrong_effort": "Wrong effort level",
    "effort_not_maintained": "Could not maintain the instructed effort",
    "incorrect_movement": "Incorrect movement (not as instructed)",
    "hand_movement": "Unintended hand / wrist movement",
    "grasp_change": "Unintended grasp change",
    "incorrect_posture": "Incorrect hand posture (e.g. fist instead of open hand)",
    "fatigue": "Fatigue or discomfort",
    "contact_problem": "Electrode contact problem",
    "other": "Other (see comment)",
}
ABORT_REASONS = {
    "user_abort": "aborted by the operator",
    "manual_stop": "manual Stop pressed during the trial",
    "user_disconnect": "device disconnected by the operator",
    "disconnected": "device disconnected",
    "link_lost": "link lost (reconnecting)",
    "stale_stream": "EMG stream stale",
    "segment_reset": "stream / segment reset",
    "recorder_error": "recorder error",
    "metadata_error": "trial.json could not be written",
    "timing_overrun": "phase timing overrun (GUI delayed)",
    "app_close": "application closed",
}
TRIAL_SCHEMA = "gforce_emg_ui.stiffness_trial"
TIME_SEMANTICS = {
    "host_mono": "time.monotonic() seconds on the acquisition host; the same timebase as "
                 "host_mono in emg.csv, imu.csv and markers.csv. Use for durations and "
                 "for selecting intervals.",
    "host_time": "time.time() wall clock (Unix seconds); for cross-referencing only, it can jump.",
    "phase_events": "When the app processed the phase transition and requested the visual "
                    "(and optional audio) cue. Display and speaker onset are NOT measured.",
    "emg_receipt": "emg.csv host_time/host_mono is the receipt time of a whole batch on the "
                   "host, not a per-sample acquisition time; samples were acquired earlier "
                   "(BLE batching, ~150 ms delivery bursts observed on our band).",
    "device_time": "SDK value derived from sample_index / nominal rate; not a calibrated "
                   "hardware clock.",
    "last_received_emg": "Segment / sample index of the last EMG batch RECEIVED when an event "
                         "was processed: a reference, not an exact physiological boundary. "
                         "Exclude windows near phase boundaries when processing.",
    "not_measured": "Sensor-to-host latency, display/audio latency and camera/robot "
                    "synchronization are not measured.",
}


@dataclass
class TrialSpec:
    participant_id: str = ""
    session_id: str = ""
    hand_side: str = "right"
    object_id: str = EMPTY_HAND
    object_name: str = "Empty hand"
    posture_id: str = "fist"
    posture_description: str = ""
    target_label: str = "low"
    motion_condition: str = STATIONARY
    dataset_role: str = "practice"
    output_root: Path = Path("data") / "stiffness_collection"
    trial_number: int = 1                    # manual selection only
    notes: str = ""
    planned_trial: Optional[dict] = None     # plan mode: plan.trials[i]
    plan_id: Optional[str] = None

    @property
    def trial_id(self) -> str:
        if self.planned_trial is not None:
            return self.planned_trial["trial_id"]
        return f"m{self.trial_number:03d}"

    @property
    def trial_type(self) -> str:
        return trial_type(self.object_id)

    def problems(self) -> List[str]:
        p = []
        if not valid_id(self.participant_id):
            p.append("Participant ID: 1-32 letters, digits, _ or - ('simulated' is reserved)")
        if not valid_id(self.session_id):
            p.append("Session ID: 1-32 letters, digits, _ or -")
        if self.hand_side not in HAND_SIDES:
            p.append("Hand side: right or left")
        if not valid_key(self.object_id):
            p.append("Object ID: lowercase letters, digits, _")
        if not valid_key(self.posture_id):
            p.append("Posture ID: lowercase letters, digits, _")
        if self.target_label not in TARGETS:
            p.append("Target: low, medium or high")
        if self.motion_condition not in MOTIONS:
            p.append(f"Movement: one of {', '.join(MOTIONS)}")
        if self.dataset_role not in ROLES:
            p.append(f"Dataset role: one of {', '.join(ROLES)}")
        task = TASKS.get(self.object_id)
        if task is not None:
            if task.stationary_only and self.motion_condition != STATIONARY:
                p.append(f"{task.short} is stationary only")
            if self.posture_id != task.posture_id:
                p.append(f"{task.short} posture is {task.posture_id}")
        if self.planned_trial is None:
            if self.dataset_role == "development":
                p.append("Development trials come from the plan (use Plan mode)")
            if not 1 <= int(self.trial_number) <= 999:
                p.append("Trial number: 1-999")
        else:
            t = self.planned_trial
            if self.dataset_role != "development" or not self.plan_id:
                p.append("Planned trials are development trials of a saved plan")
            if (t["object_id"], t["target_label"]) != (self.object_id, self.target_label):
                p.append("Condition / target do not match the planned trial")
            # v1 plans assign no motion: their trials are recorded stationary
            if t.get("motion_condition", STATIONARY) != self.motion_condition:
                p.append("Movement does not match the planned trial")
            if t.get("posture_id", self.posture_id) != self.posture_id:
                p.append("Posture does not match the planned trial")
        if not str(self.output_root).strip():
            p.append("Output directory not set")
        if len(self.notes) > 2000:
            p.append("Notes: at most 2000 characters")
        return p


AUTOMATIC, MANUAL_OVERRIDE = "automatic", "manual_override"
AVAILABILITY_BYPASS_TEXT = "Object manually selected; not in automatic available set"


@dataclass
class PendingSelection:
    """The upcoming planned trial (idle only): proposal + one-trial edits. Never saved
    into the plan; an edit changes only its own field of `requested`."""
    proposal: Optional[int]                    # automatic proposal (plan index), None = none
    proposal_selection: Optional[dict]         # how it was drawn
    requested: Dict[str, Optional[str]]        # object_id / target_label / motion_condition
    edited: List[str] = field(default_factory=list)    # fields changed by the operator
    enforced: Dict[str, Optional[str]] = field(default_factory=dict)   # forced: previous value
    resolved: Optional[int] = None             # incomplete planned trial with these values
    how: str = ""                              # proposal | lowest_index
    problem: str = ""                          # why nothing matches (Start blocked)
    warnings: List[str] = field(default_factory=list)
    availability_bypass: bool = False
    ordering_bypass: bool = False
    repeat: bool = False

    @property
    def mode(self) -> str:
        return MANUAL_OVERRIDE if self.edited else AUTOMATIC


class CollectionController:
    def __init__(self, hub, recorder, source, *, settings: Optional[ProtocolSettings] = None,
                 context: Optional[Callable[[], dict]] = None,
                 clock: Callable[[], float] = time.monotonic,
                 wall: Callable[[], float] = time.time, io: Optional[IoWorker] = None):
        self.hub = hub
        self.recorder = recorder
        self.source = source
        self.settings = settings or ProtocolSettings()
        self.context = context or (lambda: {})
        self._clock = clock
        self._wall = wall
        self.io = io or IoWorker()
        self.tracker = StreamTracker(self.settings.emg_freshness_s)
        hub.subscribe_emg(self.tracker.on_emg, name="stiffness-collection-tracker")
        self.monitor = ReadinessMonitor(self.tracker, clock)
        self.readiness: Optional[Readiness] = None
        self.on_cue: Optional[Callable[[ProtocolEvent], dict]] = None

        self.state = IDLE
        self.machine: Optional[TrialStateMachine] = None
        self.session = None
        self.spec: Optional[TrialSpec] = None          # frozen copy for the current attempt
        self.attempt_selection: Optional[dict] = None  # its selection provenance (frozen)
        self.sidecar: Optional[dict] = None
        self.sidecar_path: Optional[Path] = None
        self.metadata_ok = False
        self.message = ""
        self.error = ""
        self.last_decision: Optional[str] = None
        self.reviews_saved = 0                   # successful review writes (UI bookkeeping)
        self.stop_requests = 0
        self._writes: list = []                 # (future, kind, text)
        self._initial_write_error: Optional[str] = None
        self._review_backup = None
        self._start_segment = None
        self._git = self.io.submit(git_info)

        self.output_root: Optional[Path] = None
        self.participant_id: Optional[str] = None
        self.records: Dict[str, dict] = {}
        self.plans: List[TrialPlan] = []
        self.plan: Optional[TrialPlan] = None
        self.plan_cursor: Optional[int] = None
        self.available_objects: Optional[set] = None   # None = all plan objects
        self.selection: Optional[dict] = None          # how plan_cursor was chosen
        self.pending: Optional[PendingSelection] = None   # next trial incl. one-trial edits
        self.selection_version = 0                     # bumps when `pending` changes
        self._exclude: Optional[int] = None            # just-rejected trial, not re-picked at once
        self._discarded = 0                            # deleted attempts since the last accept
        self.progress: Optional[Progress] = None
        self.version = 0                         # bumps when records / plan / progress change
        self.objects = merge_objects(None)
        self.unreadable: List[str] = []
        self._scan = None                        # (future, participant root)
        self._plan_write = None                  # (future, plan)
        self._plan_extend = None                 # (future, plan_id, participant root)
        self._objects_write = None

    # -- properties -----------------------------------------------------------------
    @property
    def is_simulation(self) -> bool:
        return bool(getattr(self.source, "is_simulation", False))

    @property
    def data_source(self) -> str:
        return "simulation" if self.is_simulation else "real"

    @property
    def owns_recorder(self) -> bool:
        return (self.session is not None and self.recorder.session is self.session
                and not self.session.done)

    @property
    def busy(self) -> bool:
        """Something must finish before the app may close."""
        return (self.state in (RUNNING, FINALIZING, SAVING_REVIEW) or bool(self._writes)
                or self._plan_write is not None or self._plan_extend is not None
                or self._objects_write is not None)

    @property
    def loading(self) -> bool:
        return self._scan is not None

    @property
    def extending(self) -> bool:
        return self._plan_extend is not None

    @property
    def live_ids(self) -> set:
        if self.state in (RUNNING, FINALIZING) and self.sidecar:
            return {self.sidecar["attempt_id"]}
        return set()

    def planned_trial(self) -> Optional[dict]:
        """The planned trial Start would record: the proposal, or the edited match."""
        p = self.pending
        if self.plan is None or p is None or p.resolved is None:
            return None
        return self.plan.trials[p.resolved]

    def proposed_trial(self) -> Optional[dict]:
        """The automatic (random) proposal, edited or not."""
        if self.plan is None or self.plan_cursor is None:
            return None
        return self.plan.trials[self.plan_cursor]

    # -- participant, plan, objects ---------------------------------------------------
    def participant_root(self, output_root=None, participant_id=None) -> Path:
        return participant_root(output_root or self.output_root,
                                participant_id or self.participant_id, self.is_simulation)

    def open_participant(self, output_root: Path, participant_id: str) -> bool:
        """(Re)load plans, attempts and the object catalog on the I/O thread."""
        if self.state != IDLE or self.extending:
            return False
        self.output_root = Path(output_root)
        self.participant_id = participant_id if valid_id(participant_id) else None
        self.records, self.plans, self.plan = {}, [], None
        self.plan_cursor, self.progress, self.unreadable = None, None, []
        self._set_pending(None)                  # never carry a selection into another context
        if self.participant_id is None:
            self._scan = None
            return False
        root, proot = self.output_root, self.participant_root()
        fut = self.io.submit(lambda: {"objects": load_objects(root), **scan_participant(proot)})
        self._scan = (fut, proot)
        return True

    def select_plan(self, plan_id: Optional[str]):
        if self.state != IDLE or self.extending:
            return
        self.plan = next((p for p in self.plans if p.plan_id == plan_id), None)
        self._recompute()
        self._exclude, self._discarded = None, 0
        self._set_pending(None)
        self._pick_next()
        self._update_pending()

    def available(self) -> set:
        """Objects available now (empty hand is always added by the selection). Task
        conditions (surface press) only when ticked."""
        if self.plan is None:
            return set()
        objs = set(self.plan.counts) - {EMPTY_HAND}
        if self.available_objects is None:
            return objs - set(TASK_CONDITIONS)
        return objs & set(self.available_objects)

    def set_available_objects(self, objects) -> bool:
        if self.state != IDLE:
            return False
        self.available_objects = set(objects)
        self._pick_next()
        self._update_pending()                   # an edited draft keeps its values
        self.version += 1
        return True

    def _recent_targets(self) -> List[str]:
        mine = sorted((r for r in self.records.values() if r.get("plan_id") == self.plan.plan_id),
                      key=lambda r: ((r.get("timing") or {}).get("start") or {}).get("host_time") or 0)
        return [(r.get("label") or {}).get("target_label") for r in mine][-MAX_TARGET_RUN:]

    def _pick_next(self):
        """Random next planned trial among the available objects (+ empty hand)."""
        if self.plan is None or self.progress is None:
            self.plan_cursor, self.selection = None, None
            return
        avail, recent = self.available(), self._recent_targets()
        key = str(self.progress.attempted)             # new draw after every attempt
        if self._discarded:                            # deleted attempts are not on disk
            key += f"/d{self._discarded}"
        self.plan_cursor = pick_trial(self.plan, self.progress, avail, recent, key,
                                      self._exclude)
        self.selection = {"rule": SELECTION_RULE, "available_objects": sorted(avail),
                          "empty_hand_always_eligible": True,
                          "eligible": len(eligible_trials(self.plan, self.progress, avail)),
                          "recent_targets": recent, "key": key, "excluded_index": self._exclude}

    # -- pending selection (one-trial edits) -------------------------------------------
    def _set_pending(self, pending: Optional[PendingSelection]):
        self.pending = pending
        self.selection_version += 1

    def _update_pending(self):
        """Follow the proposal, or re-resolve an edited draft (its values are kept)."""
        if self.plan is None or self.progress is None:
            self._set_pending(None)
        elif self.pending is not None and self.pending.edited:
            self.resolve_selection()
        else:
            t = self.proposed_trial()
            self._set_pending(PendingSelection(
                self.plan_cursor, copy.deepcopy(self.selection),
                trial_values(self.plan, t) if t else dict.fromkeys(SELECTION_FIELDS),
                resolved=self.plan_cursor, how="proposal" if t else ""))

    def resolve_selection(self) -> Optional[int]:
        """Resolve the edited request to an incomplete planned trial (index) or a problem.

        Prefers the original proposal, then the current match, then the lowest plan
        index. Unedited fields keep the automatic rules (the object must be ticked);
        an explicitly edited object may bypass availability for this trial only.
        Nothing is consumed or completed here.
        """
        p = self.pending
        if p is None or not p.edited or self.plan is None or self.progress is None:
            return self.plan_cursor if p is not None else None
        before = (p.resolved, p.how, p.problem, list(p.warnings))
        req, avail = p.requested, self.available() | {EMPTY_HAND}
        obj = req.get("object_id")
        idx, how, problem = None, "", ""
        if obj and obj not in avail and "object_id" not in p.edited and obj in self.plan.counts:
            problem = (f"{obj} is not ticked as available: tick it, choose another object "
                       "or reset edits.")
        else:
            idx, how, problem = resolve_request(  # a kept match keeps how it was chosen
                self.plan, self.progress, req,
                [("proposal", p.proposal), (p.how or "kept", p.resolved)])
        p.availability_bypass = idx is not None and obj not in avail
        p.ordering_bypass = idx is not None and \
            run_banned_target(self._recent_targets()) == req.get("target_label")
        p.resolved, p.how, p.problem = idx, how, problem
        p.warnings = ([AVAILABILITY_BYPASS_TEXT] if p.availability_bypass else []) + (
            [f"Third {req['target_label']} target in a row (the random order avoids this)"]
            if p.ordering_bypass else [])
        if (p.resolved, p.how, p.problem, p.warnings) != before:
            self.selection_version += 1
        return idx

    def edit_selection(self, field_name: str, value: str) -> bool:
        """Set one field of the next trial (idle only); the other fields stay as they are."""
        p = self.pending
        if self.state != IDLE or p is None or self.plan is None or field_name not in SELECTION_FIELDS:
            return False
        if not {"object_id": valid_key, "target_label": TARGETS.__contains__,
                "motion_condition": MOTIONS.__contains__}[field_name](value):
            return False
        if p.requested.get(field_name) == value:
            return True
        if not p.edited:                         # remember the proposal (provenance, reset)
            p.proposal, p.proposal_selection = self.plan_cursor, copy.deepcopy(self.selection)
        p.requested[field_name] = value
        changed = {field_name}
        motion = p.requested.get("motion_condition")
        if field_name == "object_id" and value in TASKS and TASKS[value].stationary_only \
                and motion != STATIONARY:
            p.enforced["motion_condition"] = motion          # surface press: stationary only
            p.requested["motion_condition"] = STATIONARY
            changed.add("motion_condition")
        elif field_name == "object_id" and "motion_condition" in p.enforced:
            p.requested["motion_condition"] = p.enforced.pop("motion_condition")   # restore
        elif field_name == "motion_condition":
            p.enforced.pop("motion_condition", None)
        p.edited = [f for f in SELECTION_FIELDS if f in p.edited or f in changed]
        p.repeat = False
        self.selection_version += 1
        self.resolve_selection()
        return True

    def clear_edits(self) -> bool:
        """Back to automatic selection: the original proposal if still valid, else a
        normal pick by the existing selector. No-op without edits (no re-roll)."""
        p = self.pending
        if self.state != IDLE or self.plan is None or self.progress is None:
            return False
        if p is None or not p.edited:
            return True
        o = p.proposal
        if o is not None and self.plan.trials[o]["trial_id"] not in self.progress.completed \
                and self.plan.trials[o]["object_id"] in self.available() | {EMPTY_HAND}:
            self.plan_cursor, self.selection = o, p.proposal_selection
        else:
            self._pick_next()
        self._set_pending(None)
        self._update_pending()
        return True

    def differing_fields(self) -> List[str]:
        """Fields of the pending request that differ from the automatic proposal."""
        p = self.pending
        if p is None or not p.edited:
            return []
        if p.proposal is None:
            return list(p.edited)
        auto = trial_values(self.plan, self.plan.trials[p.proposal])
        return [f for f in SELECTION_FIELDS if p.requested.get(f) != auto[f]]

    def selection_record(self) -> Optional[dict]:
        """trial.json 'selection' for the pending planned trial (additive provenance)."""
        p, t = self.pending, self.planned_trial()
        if p is None or t is None:
            return None
        if not p.edited:
            return {"mode": AUTOMATIC, **copy.deepcopy(p.proposal_selection or {})}
        auto = None
        if p.proposal is not None:
            o = self.plan.trials[p.proposal]
            auto = {"trial_id": o["trial_id"], "planned_trial_index": o["index"],
                    **trial_values(self.plan, o),
                    "selection": copy.deepcopy(p.proposal_selection)}
        return {"mode": MANUAL_OVERRIDE, "rule": MANUAL_SELECTION_RULE,
                "automatic_proposal": auto,
                "edited_fields": list(p.edited), "differing_fields": self.differing_fields(),
                "enforced_fields": sorted(p.enforced),
                "requested": dict(p.requested),
                "resolved": {"trial_id": t["trial_id"], "planned_trial_index": t["index"],
                             "how": p.how},
                "available_objects": sorted(self.available()),
                "empty_hand_always_eligible": True,
                "availability_bypass": p.availability_bypass,
                "ordering_preference_bypass": p.ordering_bypass,
                "recent_targets": self._recent_targets(),
                "repeat": p.repeat}

    def _selection_blockers(self, spec: TrialSpec) -> List[str]:
        p, t = self.pending, self.planned_trial()
        b = []
        if spec.object_id not in self.available() | {EMPTY_HAND} and not (
                p is not None and p.availability_bypass and p.requested["object_id"] == spec.object_id):
            b.append(f"{spec.object_id} is not marked available")
        if t is None:
            b.append(p.problem if p is not None and p.problem else "No pending planned trial")
        elif spec.trial_id != t["trial_id"] or p.requested != {
                "object_id": spec.object_id, "target_label": spec.target_label,
                "motion_condition": spec.motion_condition}:
            b.append("Next-trial selection changed: check it and press Start again")
        return b

    def create_plan(self, seed: Optional[int] = None) -> Optional[TrialPlan]:
        """Generate a plan; it is usable only after its file was written."""
        if self.participant_id is None or self._plan_write is not None or self.state != IDLE \
                or self._plan_extend is not None:
            return None
        plan = TrialPlan.create(self.participant_id, self.data_source, seed)
        path = self.participant_root() / plan.filename
        text = dumps(plan.to_dict())

        def job():
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                raise FileExistsError(f"{path} already exists")
            atomic_write_text(path, text)
        self._plan_write = (self.io.submit(job), plan)
        return plan

    def extension_status(self, condition: str = SURFACE_PRESS) -> Tuple[str, str]:
        """Whether the selected plan can get the task condition's trials (see plan.py)."""
        if self.plan is None:
            return INCOMPATIBLE, "No plan selected."
        return extension_status(self.plan, condition)

    def extend_plan(self, condition: str = SURFACE_PRESS) -> bool:
        """Explicit, one-time extension of the selected plan file by one task condition
        (same plan_id). Idle only; plan writes, plan / participant changes and Start
        wait for it."""
        if self.state != IDLE or self.busy or self._scan is not None or self.plan is None:
            self.error = "The plan can only be extended while idle (no trial, review or save open)."
            return False
        status, why = extension_status(self.plan, condition)
        if status != EXTENDABLE:
            self.error = why
            return False
        proot = self.participant_root()
        fut = self.io.submit(extend_plan_file, proot / self.plan.filename,
                             copy.deepcopy(self.plan), None, condition)
        self._plan_extend = (fut, self.plan.plan_id, proot)
        self.error, self.message = "", f"Extending plan {self.plan.plan_id}…"
        return True

    def save_objects(self, objects) -> bool:
        if self.output_root is None or self._objects_write is not None:
            return False
        objects = merge_objects(objects)
        root, text = self.output_root, dumps(objects_doc(objects))

        def job():
            root.mkdir(parents=True, exist_ok=True)
            atomic_write_text(root / OBJECTS_FILE, text)
        self._objects_write = (self.io.submit(job), objects)
        return True

    def object_name(self, object_id: str) -> str:
        return dict(self.objects).get(object_id) or TASK_CONDITIONS.get(object_id, object_id)

    def _recompute(self):
        self.progress = (compute_progress(self.plan, self.records, self.live_ids)
                         if self.plan is not None else None)
        self.version += 1

    def role_counts(self) -> Dict[str, Dict[str, int]]:
        """Attempts per dataset role and category (all sessions of this participant)."""
        out: Dict[str, Dict[str, int]] = {}
        for rec in self.records.values():
            role = rec.get("dataset_role", "?")
            cat = attempt_category(rec, self.live_ids)
            out.setdefault(role, {}).setdefault(cat, 0)
            out[role][cat] += 1
        return out

    def next_attempt_number(self, spec: TrialSpec) -> int:
        n = 0
        for r in self.records.values():
            if r.get("trial_id") != spec.trial_id:
                continue
            if spec.planned_trial is not None:
                n += r.get("plan_id") == spec.plan_id
            else:
                n += (r.get("plan_id") is None
                      and r.get("collection_session_id") == spec.session_id
                      and r.get("dataset_role") == spec.dataset_role)
        return n + 1

    # -- readiness / start -------------------------------------------------------------
    def evaluate(self) -> Readiness:
        s = self.settings
        self.readiness = self.monitor.evaluate(self.source.state, s.readiness_s, s.emg_freshness_s)
        return self.readiness

    def start_blockers(self, spec: TrialSpec) -> List[str]:
        b = []
        if self.state == REVIEW:
            b.append("Review the previous attempt first")
        elif self.state != IDLE:
            b.append(f"Collection busy ({self.state})")
        if self.recorder.busy and not self.owns_recorder:
            b.append("Recorder busy (manual recording open or finalizing)")
        b += spec.problems()
        b += self.settings.validate()
        if self._scan is not None:
            b.append("Loading participant data…")
        if self._plan_extend is not None:
            b.append("Plan update in progress")
        elif (self.participant_id != spec.participant_id or self.output_root is None
              or Path(spec.output_root) != self.output_root):
            b.append("Participant data not loaded")
        if spec.planned_trial is not None:
            if self.plan is None or self.plan.plan_id != spec.plan_id:
                b.append("No saved plan selected")
            elif self.progress and spec.trial_id in self.progress.completed:
                b.append(f"Planned trial {spec.trial_id} is already completed")
            else:                                # shown == started == recorded == counted
                b += self._selection_blockers(spec)
        r = self.readiness or self.evaluate()
        if not r.ready:
            b += r.blockers
        return b

    def start_trial(self, spec: TrialSpec) -> bool:
        self.error = ""
        self.evaluate()
        if spec.planned_trial is not None and self.state == IDLE:
            self._update_pending()               # resolve again now, against the same state
        blockers = self.start_blockers(spec)
        if blockers:
            self.error = "Cannot start: " + "; ".join(blockers)
            return False
        spec = copy.deepcopy(spec)               # frozen for this attempt
        selection = self.selection_record() if spec.planned_trial is not None else None
        target = TARGETS[spec.target_label]
        attempt_id = uuid.uuid4().hex[:8]
        number = self.next_attempt_number(spec)
        session_root = participant_root(spec.output_root, spec.participant_id,
                                        self.is_simulation) / spec.session_id
        try:
            name = attempt_name(spec.object_id, spec.target_label, target["s"], spec.trial_id,
                                number, attempt_id, spec.dataset_role)
            Path(spec.output_root).mkdir(parents=True, exist_ok=True)
            ensure_within(spec.output_root, session_root)
            machine = TrialStateMachine(self.settings, self._clock, self._wall)
        except (ValueError, OSError) as e:
            self.error = f"Cannot start: {e}"
            return False
        ctx = self.context() or {}
        meta = {"device": ctx.get("device", {}), "emg_rate": self.hub.emg_rate,
                "imu_rate": self.hub.imu_rate,
                "onboard_filters": (ctx.get("params") or {}).get("filters", {}),
                "stiffness_collection": {"attempt_id": attempt_id, "trial_id": spec.trial_id,
                                         "sidecar": "trial.json",
                                         "data_source": self.data_source}}
        self.tracker.start_capture()
        try:
            session = self.recorder.start(name, meta, root=session_root)
        except Exception as e:
            self.tracker.stop_capture()
            self.error = f"Could not start recording: {e}"
            return False
        self.session, self.machine, self.spec = session, machine, spec
        self.attempt_selection = selection
        self.state = RUNNING
        self.metadata_ok = False
        self.message = ""
        self.last_decision = None
        self._initial_write_error = None
        self._start_segment = self.readiness.segment
        self.sidecar = self._new_sidecar(spec, attempt_id, number, session, ctx)
        self.sidecar_path = session.folder / "trial.json"
        events = machine.start()
        self.sidecar["timing"]["start"] = {"host_mono": events[0].host_mono,
                                           "host_time": events[0].host_time}
        self._handle(events)
        self._write_sidecar("initial")        # identity + status "recording", persisted early
        self._recompute()
        return True

    # -- running ---------------------------------------------------------------------
    def tick(self):
        """Call periodically from the GUI thread (e.g. every 50 ms)."""
        self._poll_io()
        self.evaluate()
        if self.state == RUNNING:
            reason = self._running_problem()
            if reason:
                self.abort(reason)
            else:
                self._handle(self.machine.update())
                if self.machine.consume_stop_request():
                    self._request_stop()
        if self.state == FINALIZING and self.session.done and \
                self.sidecar.get("status") == "recording":
            self._finalize()

    def _running_problem(self) -> Optional[str]:
        state = self.source.state
        if state != STREAMING:
            return "link_lost" if state == RECONNECTING else "disconnected"
        last = self.tracker.last
        if last is None or self._clock() - last.host_mono > self.settings.emg_freshness_s:
            return "stale_stream"
        if last.segment != self._start_segment:
            return "segment_reset"
        if self.session.state == ERROR:
            return "recorder_error"
        if self._initial_write_error:
            return "metadata_error"
        return None

    def abort(self, reason: str, detail: Optional[dict] = None) -> bool:
        if self.state != RUNNING or self.machine is None:
            return False
        if reason == "recorder_error":
            detail = {"recorder_error": self.session.error, **(detail or {})}
        elif reason == "metadata_error":
            detail = {"error": self._initial_write_error, **(detail or {})}
        self._handle(self.machine.abort(reason, detail))
        self.message = f"Aborted: {ABORT_REASONS.get(reason, reason)}"
        if self.machine.consume_stop_request():
            self._request_stop()
        return True

    def extend_rest(self) -> bool:
        if self.state != RUNNING:
            return False
        events = self.machine.extend_rest()
        self._handle(events)
        return bool(events)

    def _handle(self, events: List[ProtocolEvent]):
        last = self.tracker.last
        for ev in events:
            if self.session is not None:
                self.session.add_marker(ev.name, ev.host_time, ev.host_mono)
            cue = {}
            if self.on_cue is not None:
                try:
                    cue = self.on_cue(ev) or {}
                except Exception as e:            # a cue failure never stops collection
                    cue = {"error": f"{type(e).__name__}: {e}"}
            self.sidecar["timing"]["events"].append({
                "name": ev.name, "host_mono": ev.host_mono, "host_time": ev.host_time,
                "phase_after": ev.phase, "detail": ev.detail, "cue": cue,
                "last_received_emg": None if last is None else {
                    "segment": last.segment, "sample_index": last.last_index,
                    "receipt_host_mono": last.host_mono}})

    def _request_stop(self):
        now, wall = self._clock(), self._wall()
        self.sidecar["timing"]["stop_requested"] = {"host_mono": now, "host_time": wall}
        self.sidecar["acquisition"]["received_rates_at_stop_hz"] = {
            "emg": self.hub.emg_actual_rate(), "imu": self.hub.imu_actual_rate()}
        self.stop_requests += 1
        self.session.request_stop()
        self.state = FINALIZING

    # -- finalize / review ---------------------------------------------------------------
    def _finalize(self):
        s, m, sc, st = self.session, self.machine, self.sidecar, self.settings
        prev, records, truncated = self.tracker.stop_capture()
        completed = m.phase == DONE
        sc["status"] = "finalized"
        sc["timing"].update({
            "end": {"host_mono": m.end_mono,
                    "host_time": m.events[-1].host_time if m.events else None},
            "protocol_duration_s": m.elapsed(), "recorder_duration_s": s.elapsed,
            "phases": [dict(iv) for iv in m.intervals], "flags": list(m.flags),
            "rest_extensions": m.rest_extensions})
        sc["outcome"] = {
            "protocol": "completed" if completed else "aborted",
            "abort_reason": m.abort_reason,
            "abort_reason_text": ABORT_REASONS.get(m.abort_reason) if m.abort_reason else None,
            "recorder": "saved" if s.state == SAVED else "error",
            "recorder_error": s.error, "metadata": "ok"}
        sc["recording"]["recorder_state"] = s.state
        sc["recording"]["recorder_counts"] = {k: dict(v) for k, v in s.counts.items()}
        whole = summarize_window(records, prev, m.start_mono, m.end_mono, st.emg_freshness_s)
        hold = next((iv for iv in m.intervals if iv["phase"] == HOLD), None)
        hold_done = hold is not None and hold["completed"]
        cand = {"phase": TRAINING_CANDIDATE_PHASE, "start_marker": "HOLD_START",
                "end_marker": "HOLD_END", "eligible": False, "quality_status": "not_available",
                "quality_flags": [], "start_host_mono": None, "end_host_mono": None,
                "motion_condition": self.spec.motion_condition,
                "note": "Candidate interval only: it does not assert that the participant "
                        "achieved the instructed effort. PREPARE, SETTLE and REST are never "
                        "training data for the target. HOLD = maintain the instructed effort "
                        "(during the instructed movement, if any), not 'stay still'."}
        hold_q = None
        if hold_done:
            t0, t1 = hold["start_host_mono"], hold["end_host_mono"]
            hold_q = summarize_window(records, prev, t0, t1 + st.hold_quality_tail_s,
                                      st.emg_freshness_s)
            flags = list(hold_q["flags"])
            flags += [f"timing_{f['type']}:{f['phase']}" for f in m.flags if f["phase"] == HOLD]
            if s.state != SAVED:
                flags.append("recorder_error")
            if s.counts["emg"]["dropped"]:
                flags.append(f"recorder_dropped:{s.counts['emg']['dropped']}")
            if truncated:
                flags.append("quality_capture_truncated")
            if not completed:
                flags.append(f"protocol_aborted:{m.abort_reason}")
            cand.update(start_host_mono=t0, end_host_mono=t1,
                        start_host_time=hold["start_host_time"],
                        end_host_time=hold["end_host_time"], duration_s=hold["actual_s"],
                        planned_s=hold["planned_s"], quality_flags=sorted(set(flags)),
                        quality_status="clean" if not flags else "flagged",
                        eligible=completed and s.state == SAVED,
                        quality_window_tail_s=st.hold_quality_tail_s)
        sc["training_candidate"] = cand
        sc["quality"] = {"trial": whole, "hold": hold_q,
                         "contact_events": contact_events(records, prev),
                         "capture_truncated": truncated}
        self.monitor.reset()                  # the next attempt re-passes readiness
        self._write_sidecar("final")

    @property
    def accept_blockers(self) -> List[str]:
        if self.state != REVIEW or not self.sidecar:
            return ["no attempt to review"]
        out, sc = [], self.sidecar
        if not self.metadata_ok:
            out.append("trial.json was not written")
        if sc["outcome"]["protocol"] != "completed":
            out.append(f"protocol {sc['outcome']['protocol']} ({sc['outcome']['abort_reason']})")
        if sc["outcome"]["recorder"] != "saved":
            out.append("recording not saved")
        cand = sc.get("training_candidate") or {}
        if cand.get("quality_status") != "clean":
            out.append("HOLD quality " + cand.get("quality_status", "unknown") +
                       (": " + ", ".join(cand.get("quality_flags", [])) if cand.get("quality_flags") else ""))
        if self.spec and self.spec.planned_trial is not None and self.progress \
                and self.spec.trial_id in self.progress.completed:
            out.append("planned trial already completed")
        return out

    @property
    def acceptable(self) -> bool:
        return not self.accept_blockers

    def submit_review(self, decision: str, reasons=(), comment: str = "") -> bool:
        if self.state != REVIEW or decision not in ("accept", "reject", "repeat"):
            return False
        if decision == "accept" and not self.acceptable:
            self.error = "Cannot accept: " + "; ".join(self.accept_blockers)
            return False
        unknown = [r for r in reasons if r not in REVIEW_REASONS]
        if unknown:
            raise ValueError(f"unknown review reasons {unknown}")
        self.error = ""
        sc = self.sidecar
        self._review_backup = (json.loads(json.dumps(sc["review"])), sc["status"])
        sc["review"] = {"status": "accepted" if decision == "accept" else "rejected",
                        "decision": decision, "reasons": list(reasons),
                        "comment": comment.strip(), "reviewed_host_time": self._wall(),
                        "reviewed_host_mono": self._clock(), "reviewer": "operator"}
        sc["status"] = "reviewed"
        self.last_decision = decision
        self.state = SAVING_REVIEW
        if decision == "accept":
            self._write_sidecar("review")
        else:                                 # only accepted attempts are kept
            text = dumps(sc)
            fut = self.io.submit(discard_attempt, self.spec.output_root, self.session.folder, text)
            self._writes.append((fut, "discard", text))
        return True

    # -- I/O ---------------------------------------------------------------------------
    def _write_sidecar(self, kind: str):
        text = dumps(self.sidecar)
        fut = self.io.submit(atomic_write_text, self.sidecar_path, text)
        self._writes.append((fut, kind, text))

    def _upsert(self, text: str):
        rec = json.loads(text)
        rec["_path"] = str(self.sidecar_path)
        self.records[rec["attempt_id"]] = rec

    def _poll_io(self):
        while self._writes and self._writes[0][0].done():
            fut, kind, text = self._writes.pop(0)
            exc = fut.exception()
            err = None if exc is None else f"{type(exc).__name__}: {exc}"
            if kind == "initial":
                if err:
                    self._initial_write_error = err
                else:
                    self._upsert(text)
            elif kind == "final":
                self._final_written(err, text)
            elif kind == "review":
                self._review_written(err, text)
            elif kind == "discard":
                self._attempt_discarded(err, text)
            self._recompute()
        if self._scan is not None and self._scan[0].done():
            fut, proot = self._scan
            self._scan = None
            if self.participant_id is not None and proot == self.participant_root():
                try:
                    res = fut.result()
                except Exception as e:
                    self.error = f"Could not read participant data: {e}"
                else:
                    self.objects = res["objects"]
                    self.plans = res["plans"]
                    self.records.update(res["records"])
                    self.unreadable = res["unreadable"]
                    self.select_plan(self.plans[0].plan_id if self.plans else None)
        if self._plan_write is not None and self._plan_write[0].done():
            fut, plan = self._plan_write
            self._plan_write = None
            if fut.exception() is not None:
                self.error = f"Plan was NOT saved: {fut.exception()}"
            elif plan.participant_id == self.participant_id:
                self.error = ""
                self.plans.insert(0, plan)
                self.select_plan(plan.plan_id)
                self.message = f"Plan {plan.plan_id} saved ({plan.total} trials, seed {plan.seed})"
        if self._plan_extend is not None and self._plan_extend[0].done():
            fut, plan_id, proot = self._plan_extend
            self._plan_extend = None
            if fut.exception() is not None:
                self.error = f"Plan was NOT extended (saved plan unchanged): {fut.exception()}"
            else:
                new = fut.result()
                if self.participant_id is not None and proot == self.participant_root():
                    self.plans = [new if p.plan_id == plan_id else p for p in self.plans]
                    if self.plan is not None and self.plan.plan_id == plan_id:
                        self.select_plan(plan_id)
                ext = new.extensions[-1]
                task = EXTENSIONS[ext["id"]]
                self.error = ""
                self.message = (f"Plan {plan_id}: {ext['total_before']} → {ext['total_after']} "
                                f"trials ({task.total} {task.short} added; backup {ext['backup']})")
        if self._objects_write is not None and self._objects_write[0].done():
            fut, objects = self._objects_write
            self._objects_write = None
            if fut.exception() is not None:
                self.error = f"Object names were NOT saved: {fut.exception()}"
            else:
                self.objects = objects

    def _final_written(self, err: Optional[str], text: str):
        sc, out = self.sidecar, self.sidecar["outcome"]
        if err:
            self.metadata_ok = False
            out["metadata"] = f"error: {err}"
            self.error = f"ERROR: trial.json could not be written ({err}); raw files kept in " \
                         f"{self.session.folder}. Not counted."
        else:
            self.metadata_ok = True
            self._upsert(text)
        if out["recorder"] != "saved":
            self.error = (f"ERROR: recording failed ({out['recorder_error']}); partial data kept in "
                          f"{self.session.folder}. Not counted.")
        elif self.metadata_ok and out["protocol"] == "completed":
            q = sc["training_candidate"]["quality_status"]
            self.message = f"Saved {self.session.folder.name} · HOLD quality {q}"
        elif self.metadata_ok:
            self.message = (f"Aborted ({ABORT_REASONS.get(out['abort_reason'], out['abort_reason'])})"
                            f" – not counted; Reject / Repeat deletes it.")
        self.state = REVIEW

    def _review_written(self, err: Optional[str], text: str):
        if err:
            review, status = self._review_backup
            self.sidecar["review"], self.sidecar["status"] = review, status
            self.error = f"Review was NOT saved ({err}); try again."
            self.state = REVIEW
            return
        self._upsert(text)
        self._reviewed()

    def _attempt_discarded(self, err: Optional[str], text: str):
        folder = self.session.folder
        if err:
            self._upsert(text)                # left on disk, marked rejected
        else:
            self.records.pop(self.sidecar["attempt_id"], None)
        self._reviewed()
        if err:
            self.error = f"ERROR: could not delete {folder} ({err}); delete it by hand."

    def _reviewed(self):
        decision = self.last_decision
        self._discarded = 0 if decision == "accept" else self._discarded + 1
        self.reviews_saved += 1
        self.state = IDLE
        self.session = None
        self.machine = None
        self._recompute()
        if self.spec.planned_trial is not None and self.plan is not None and self.progress:
            done = next(i for i, t in enumerate(self.plan.trials)
                        if t["trial_id"] == self.spec.trial_id)
            p = self.pending
            if decision == "repeat" and self.spec.trial_id not in self.progress.completed:
                if p is not None and p.edited:          # same edited combination and trial
                    p.resolved, p.repeat = done, True
                    self.resolve_selection()
                    self.selection_version += 1
                else:
                    self.plan_cursor = done             # explicit retry of the same trial
                    self.selection = {**(self.selection or {}), "repeat": True}
                    self._update_pending()
            else:                                       # one-shot edits end here
                self._exclude = done if decision == "reject" else None
                self._set_pending(None)
                self._pick_next()
                self._update_pending()
        self.message = {"accept": "Accepted", "reject": "Rejected – deleted",
                        "repeat": "Rejected – deleted; repeat the same trial"}[decision]

    # -- sidecar ---------------------------------------------------------------------
    def _new_sidecar(self, spec: TrialSpec, attempt_id: str, number: int, session,
                     ctx: dict) -> dict:
        r = self.readiness
        params = ctx.get("params") or {}
        git = self._git.result() if self._git.done() else {"revision": None, "dirty": None}
        st = self.settings
        return {
            "schema": TRIAL_SCHEMA, "schema_version": SCHEMA_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "software": {"app": "gforce-emg-ui", "app_version": __version__, "git": git},
            "status": "recording",
            "attempt_id": attempt_id, "attempt_number": number, "trial_id": spec.trial_id,
            "plan_id": spec.plan_id,
            "planned_trial_index": spec.planned_trial["index"] if spec.planned_trial else None,
            "selection": copy.deepcopy(self.attempt_selection) if spec.planned_trial else None,
            "participant_id": spec.participant_id, "collection_session_id": spec.session_id,
            "dataset_role": spec.dataset_role, "data_source": self.data_source,
            "is_simulation": self.is_simulation,
            "condition": {"object_id": spec.object_id, "object_name": spec.object_name,
                          "trial_type": spec.trial_type,
                          "motion_condition": spec.motion_condition,
                          "posture_id": spec.posture_id,
                          "posture_description": spec.posture_description,
                          "hand_side": spec.hand_side},
            "notes": spec.notes,
            "motion": motion_info(spec.motion_condition),
            "label": {**target_info(spec.target_label),
                      "semantics": LABEL_SEMANTICS,
                      "mapping": {k: {"target_class_id": v["class_id"], "target_s": v["s"]}
                                  for k, v in TARGETS.items()},
                      **task_instructions(spec.target_label, spec.motion_condition,
                                          spec.trial_type)},
            "protocol": {"settings": st.to_dict(), "phase_order": list(PHASES),
                         "planned_durations_s": dict(st.durations()),
                         "planned_total_s": st.total_s,
                         "training_candidate_phase": TRAINING_CANDIDATE_PHASE,
                         "markers": "markers.csv labels PREPARE_START, SETTLE_START, HOLD_START,"
                                    " HOLD_END, REST_START, REST_END, REST_EXTENDED, ABORT, "
                                    "TRIAL_END (same host_time/host_mono as timing.events)"},
            "time_semantics": TIME_SEMANTICS,
            "recording": {"folder": session.folder.name,
                          "recorder_session_id": session.session_id,
                          "files": ["emg.csv", "imu.csv", "contact.csv", "markers.csv",
                                    "session.json", "trial.json"]},
            "acquisition": {
                "device": ctx.get("device", {}),
                "backend": self.source.backend_name(),
                "emg_rate_requested_hz": getattr(self.source, "target_emg_rate", None),
                "emg_rate_readback_hz": params.get("emg_rate"),
                "emg_rate_reported_by_stream_hz": self.hub.emg_rate,
                "imu_rate_reported_by_stream_hz": self.hub.imu_rate,
                "received_rates_at_start_hz": {"emg": self.hub.emg_actual_rate(),
                                               "imu": self.hub.imu_actual_rate()},
                "rate_note": "received rate is host throughput, not a verified acquisition "
                             "rate (~505 of nominal 1000 Hz observed on FW V1.0.4)",
                "onboard_filters_readback": params.get("filters", {}),
                "emg_channel_order": [f"ch{i + 1}" for i in range(EMG_CHANNELS)],
                "emg_units": "unknown (SDK 'data' field; not verified)",
                "impedance_units": "ohm as reported by the SDK (latest cached value; "
                                   "measurement time/age not exposed by the SDK)",
                "imu_channel_order": list(IMU_LABELS), "imu_units": "unknown",
            },
            "readiness": {"settings": {"readiness_s": st.readiness_s,
                                       "emg_freshness_s": st.emg_freshness_s,
                                       "contact_rule": "not checked: electrode contact never "
                                                       "blocks Start or flags HOLD; levels and "
                                                       "impedance are recorded for reference"},
                          "initial_snapshot": r.snapshot() if r else None},
            "timing": {"start": None, "end": None, "events": [], "phases": [], "flags": []},
            "training_candidate": None,
            "quality": None,
            "outcome": {"protocol": "in_progress", "abort_reason": None,
                        "recorder": "recording", "recorder_error": "", "metadata": "pending"},
            "review": {"status": "pending"},
        }
