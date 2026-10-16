"""Development trial plan, attempt index and progress (Qt-free).

A planned trial (plan.trials[i], ID t001..t200) and a recording attempt (one
folder + trial.json, unique attempt_id) are different things: repeating a trial
creates a new attempt linked to the same planned trial, never overwriting one.

Progress is always recomputed from the trial.json files on disk (plus the
attempt currently being recorded): a planned trial counts once, only when a
real (or, in a simulated plan, simulated) attempt with matching plan, condition
and target was recorded completely, saved, passed the HOLD quality checks and
was explicitly accepted by the operator.

Plans v2 also assign a motion condition to every planned trial (each motion in
every object x target cell). v1 plan files have no motion: they are loaded as
they are, never rewritten, and their remaining trials are recorded stationary.
Recorded trials without motion_condition keep it unknown (None).
"""

import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .protocol import MOTION_INSTRUCTION_VERSION, MOTION_ORDER, MOTIONS, STATIONARY, TARGET_ORDER
from .storage import read_json, valid_key

EMPTY_HAND = "empty_hand"
DEFAULT_OBJECTS = (
    (EMPTY_HAND, "Empty hand"),
    ("obj01", "Object 1"), ("obj02", "Object 2"), ("obj03", "Object 3"),
    ("obj04", "Object 4"), ("obj05", "Object 5"),
)
DEFAULT_POSTURES = {EMPTY_HAND: ("fist", "Closed fist, no object")}
OBJECT_POSTURE = ("grasp", "Grasp the object")
OBJECTS_FILE = "objects.json"

# Low / Medium / High accepted development trials per condition (total 67/67/66 = 200)
DEFAULT_PLAN_COUNTS = {
    EMPTY_HAND: (13, 13, 14),
    "obj01": (11, 11, 10),
    "obj02": (11, 11, 10),
    "obj03": (11, 11, 10),
    "obj04": (11, 10, 11),
    "obj05": (10, 11, 11),
}
MAX_TARGET_RUN = 2
PLAN_MOTIONS = MOTION_ORDER
V1_PLAN_MOTION = STATIONARY      # v1 plans predate movement: new attempts are stationary
PLAN_GENERATOR = "weighted-random-order/no-target-run-over-2/v1"
PLAN_GENERATOR_V2 = "weighted-random-order/no-target-run-over-2/motion-split/v2"
PLAN_SCHEMA = "gforce_emg_ui.stiffness_plan"
SIMULATED_DIR = "simulated"


def default_posture(object_id: str) -> Tuple[str, str]:
    return DEFAULT_POSTURES.get(object_id, OBJECT_POSTURE)


def participant_root(output_root: Path, participant_id: str, simulated: bool) -> Path:
    """Real: <root>/<P>; simulated: <root>/simulated/<P> (kept apart from real data)."""
    root = Path(output_root)
    return (root / SIMULATED_DIR / participant_id) if simulated else (root / participant_id)


def _weighted_order(cells: Dict[tuple, int], seed: int, max_run: int) -> List[tuple]:
    """Random order of cells (cell[1] = target) weighted by remaining counts, with
    no run of more than max_run identical targets. Deterministic for a seed."""
    rng = random.Random(seed)
    keys = [c for c, n in cells.items() if n > 0]
    total = sum(cells.values())
    for _ in range(10_000):
        left = dict(cells)
        order: List[tuple] = []
        while len(order) < total:
            tail = [c[1] for c in order[-max_run:]]
            banned = tail[0] if len(tail) == max_run and len(set(tail)) == 1 else None
            options = [c for c in keys if left[c] > 0 and c[1] != banned]
            if not options:
                break                           # dead end: restart (deterministic)
            pick = rng.choices(options, weights=[left[c] for c in options])[0]
            left[pick] -= 1
            order.append(pick)
        if len(order) == total:
            return order
    raise RuntimeError("could not generate a plan satisfying the run constraint")


def generate_order(counts: Dict[str, Tuple[int, int, int]], seed: int,
                   max_run: int = MAX_TARGET_RUN) -> List[Tuple[str, str]]:
    """v1: (object_id, target) order (same sequence as before for a given seed)."""
    cells = {(obj, TARGET_ORDER[i]): counts[obj][i] for obj in sorted(counts) for i in range(3)}
    return _weighted_order(cells, seed, max_run)


def motion_split(counts: Dict[str, Tuple[int, int, int]],
                 motions: Tuple[str, ...] = PLAN_MOTIONS) -> Dict[Tuple[str, str, str], int]:
    """Split every (object, target) count over the motions as evenly as possible.

    Remainders rotate over the motions, starting at the object's index, so each
    object's motion totals differ by at most 1 and motion x target totals over the
    plan stay balanced (no motion/label confound). Cell, total and class counts
    are unchanged.
    """
    out, k = {}, len(motions)
    for n_obj, (obj, c) in enumerate(counts.items()):
        ptr = n_obj % k
        for i, tgt in enumerate(TARGET_ORDER):
            base, extra = divmod(c[i], k)
            for m in motions:
                out[(obj, tgt, m)] = base
            for e in range(extra):
                out[(obj, tgt, motions[(ptr + e) % k])] += 1
            ptr = (ptr + extra) % k
    return out


def generate_motion_order(counts: Dict[str, Tuple[int, int, int]], seed: int,
                          motions: Tuple[str, ...] = PLAN_MOTIONS,
                          max_run: int = MAX_TARGET_RUN) -> List[Tuple[str, str, str]]:
    """v2: (object_id, target, motion) order; same target-run rule as v1."""
    split = motion_split(counts, motions)
    cells = {(obj, tgt, m): split[(obj, tgt, m)]
             for obj in sorted(counts) for tgt in TARGET_ORDER for m in motions}
    return _weighted_order(cells, seed, max_run)


def max_target_run(order: Iterable[tuple]) -> int:
    best = run = 0
    prev = None
    for cell in order:
        t = cell[1]
        run = run + 1 if t == prev else 1
        prev = t
        best = max(best, run)
    return best


@dataclass
class TrialPlan:
    plan_id: str
    participant_id: str
    data_source: str                       # "real" or "simulation"
    seed: int
    created: str
    counts: Dict[str, Tuple[int, int, int]]
    trials: List[dict] = field(default_factory=list)
    motions: Optional[Tuple[str, ...]] = None   # None: v1 plan (no motion assigned)

    @classmethod
    def create(cls, participant_id: str, data_source: str, seed: Optional[int] = None,
               counts: Optional[dict] = None,
               motions: Tuple[str, ...] = PLAN_MOTIONS) -> "TrialPlan":
        counts = {k: tuple(v) for k, v in (counts or DEFAULT_PLAN_COUNTS).items()}
        seed = random.SystemRandom().randrange(1, 2 ** 31) if seed is None else int(seed)
        trials = []
        for i, (obj, tgt, motion) in enumerate(generate_motion_order(counts, seed, motions),
                                               start=1):
            posture, _ = default_posture(obj)
            trials.append({"index": i, "trial_id": f"t{i:03d}", "object_id": obj,
                           "posture_id": posture, "target_label": tgt,
                           "motion_condition": motion})
        return cls(plan_id=f"plan-{uuid.uuid4().hex[:8]}", participant_id=participant_id,
                   data_source=data_source, seed=seed,
                   created=datetime.now().isoformat(timespec="seconds"),
                   counts=counts, trials=trials, motions=tuple(motions))

    @property
    def schema_version(self) -> int:
        return 1 if self.motions is None else 2

    def motion_of(self, trial: dict) -> str:
        """Motion to instruct for a planned trial (v1 plans: stationary)."""
        return trial.get("motion_condition") or V1_PLAN_MOTION

    @property
    def filename(self) -> str:
        return f"plan_{self.plan_id}.json"

    @property
    def total(self) -> int:
        return len(self.trials)

    def trial(self, trial_id: str) -> Optional[dict]:
        for t in self.trials:
            if t["trial_id"] == trial_id:
                return t
        return None

    def to_dict(self) -> dict:
        d = {"schema": PLAN_SCHEMA, "schema_version": self.schema_version,
             "plan_id": self.plan_id, "participant_id": self.participant_id,
             "data_source": self.data_source, "dataset_role": "development", "seed": self.seed,
             "generator": PLAN_GENERATOR if self.motions is None else PLAN_GENERATOR_V2,
             "max_target_run": MAX_TARGET_RUN, "created": self.created,
             "counts_low_medium_high": {k: list(v) for k, v in self.counts.items()}}
        if self.motions is not None:
            split = motion_split(self.counts, self.motions)
            d["motion_conditions"] = list(self.motions)
            d["motion_instruction_version"] = MOTION_INSTRUCTION_VERSION
            d["motion_counts"] = {obj: {t: {m: split[(obj, t, m)] for m in self.motions}
                                        for t in TARGET_ORDER} for obj in self.counts}
        d.update(total=self.total, trials=self.trials)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TrialPlan":
        if d.get("schema") != PLAN_SCHEMA:
            raise ValueError("not a stiffness plan file")
        version = d.get("schema_version", 1)
        if version not in (1, 2):
            raise ValueError(f"unsupported plan schema_version {version}")
        motions = None
        if version == 2:
            motions = tuple(d["motion_conditions"])
            unknown = {t.get("motion_condition") for t in d["trials"]} - set(MOTIONS)
            if set(motions) - set(MOTIONS) or unknown:
                raise ValueError(f"unknown motion condition(s) in plan: {sorted(map(str, unknown))}")
        return cls(plan_id=d["plan_id"], participant_id=d["participant_id"],
                   data_source=d["data_source"], seed=int(d["seed"]), created=d["created"],
                   counts={k: tuple(v) for k, v in d["counts_low_medium_high"].items()},
                   trials=list(d["trials"]), motions=motions)


# -- attempts on disk ------------------------------------------------------------

def scan_participant(proot: Path) -> dict:
    """Plans (newest first) and attempt records under one participant folder.

    Runs on the I/O thread. Unreadable files are listed, never guessed at.
    """
    proot = Path(proot)
    plans, records, unreadable = [], {}, []
    if proot.is_dir():
        for p in sorted(proot.glob("plan_*.json")):
            try:
                plans.append(TrialPlan.from_dict(read_json(p)))
            except Exception as e:
                unreadable.append(f"{p}: {e}")
        for p in sorted(proot.glob("*/*/trial.json")):
            try:
                rec = read_json(p)
                rec["_path"] = str(p)
                records[rec["attempt_id"]] = rec
            except Exception as e:
                unreadable.append(f"{p}: {e}")
    plans.sort(key=lambda pl: pl.created, reverse=True)
    return {"plans": plans, "records": records, "unreadable": unreadable}


def attempt_category(rec: dict, live: Iterable[str] = ()) -> str:
    """accepted | rejected | unreviewed | invalid | in_progress.

    invalid = aborted, recorder/metadata error, or interrupted (trial.json never
    finalized, e.g. after a crash). Invalid attempts are never accepted.
    """
    if rec.get("attempt_id") in set(live):
        return "in_progress"
    if rec.get("status") not in ("finalized", "reviewed"):
        return "invalid"
    out = rec.get("outcome") or {}
    if out.get("protocol") != "completed" or out.get("recorder") != "saved" \
            or out.get("metadata") != "ok":
        return "invalid"
    review = (rec.get("review") or {}).get("status")
    if review == "accepted":
        return "accepted"
    if review == "rejected":
        return "rejected"
    return "unreviewed"


def motion_of(rec: dict) -> Optional[str]:
    """Recorded motion condition; None (unknown) for trials recorded before it existed."""
    return (rec.get("condition") or {}).get("motion_condition")


def counts_toward_plan(rec: dict, plan: TrialPlan) -> bool:
    planned = plan.trial(rec.get("trial_id") or "")
    cond = rec.get("condition") or {}
    label = rec.get("label") or {}
    # v2 plans: the instructed motion must match; v1 plans assign none (old records: unknown)
    motion_ok = planned is not None and (
        "motion_condition" not in planned or motion_of(rec) == planned["motion_condition"])
    return (planned is not None and motion_ok and rec.get("plan_id") == plan.plan_id
            and rec.get("data_source") == plan.data_source
            and rec.get("is_simulation") == (plan.data_source == "simulation")
            and rec.get("participant_id") == plan.participant_id
            and rec.get("dataset_role") == "development"
            and cond.get("object_id") == planned["object_id"]
            and label.get("target_label") == planned["target_label"]
            and (rec.get("training_candidate") or {}).get("quality_status") == "clean"
            and attempt_category(rec) == "accepted")


@dataclass
class Progress:
    total_planned: int = 0
    accepted: int = 0
    remaining: int = 0
    attempted: int = 0
    rejected: int = 0
    invalid: int = 0
    unreviewed: int = 0
    in_progress: int = 0
    by_cell: Dict[Tuple[str, str], List[int]] = field(default_factory=dict)  # [accepted, planned]
    by_motion: Dict[str, List[int]] = field(default_factory=dict)            # v2 plans only
    completed: Dict[str, str] = field(default_factory=dict)                  # trial_id -> attempt
    attempts_by_trial: Dict[str, int] = field(default_factory=dict)


def compute_progress(plan: TrialPlan, records: Dict[str, dict], live: Iterable[str] = ()) -> Progress:
    live = set(live)
    pr = Progress(total_planned=plan.total)
    for obj, c in plan.counts.items():
        for i, tgt in enumerate(TARGET_ORDER):
            pr.by_cell[(obj, tgt)] = [0, c[i]]
    pr.by_motion = {m: [0, 0] for m in plan.motions or ()}
    for t in plan.trials:
        if "motion_condition" in t:
            pr.by_motion.setdefault(t["motion_condition"], [0, 0])[1] += 1
    mine = [r for r in records.values() if r.get("plan_id") == plan.plan_id
            and r.get("data_source") == plan.data_source]
    for rec in sorted(mine, key=lambda r: (r.get("timing") or {}).get("start", {})
                      .get("host_time") or 0):
        tid = rec.get("trial_id")
        pr.attempts_by_trial[tid] = pr.attempts_by_trial.get(tid, 0) + 1
        pr.attempted += 1
        cat = attempt_category(rec, live)
        if cat == "accepted":
            if counts_toward_plan(rec, plan) and tid not in pr.completed:
                pr.completed[tid] = rec["attempt_id"]
                planned = plan.trial(tid)
                pr.by_cell[(planned["object_id"], planned["target_label"])][0] += 1
                if "motion_condition" in planned:
                    pr.by_motion[planned["motion_condition"]][0] += 1
        elif cat == "rejected":
            pr.rejected += 1
        elif cat == "invalid":
            pr.invalid += 1
        elif cat == "unreviewed":
            pr.unreviewed += 1
        else:
            pr.in_progress += 1
    pr.accepted = len(pr.completed)
    pr.remaining = plan.total - pr.accepted
    return pr


def next_incomplete(plan: TrialPlan, progress: Progress, after: Optional[int] = None) -> Optional[int]:
    """0-based index of the next planned trial not yet completed, searching
    forward from after+1 and wrapping around (after=None: from the start)."""
    n = plan.total
    start = 0 if after is None else after + 1
    for k in range(n):
        i = (start + k) % n
        if plan.trials[i]["trial_id"] not in progress.completed:
            return i
    return None


def training_candidate(rec: dict, require_accepted: bool = True) -> Optional[dict]:
    """The HOLD interval of an attempt, with its instructed label, or None.

    Only HOLD is ever returned, so PREPARE/SETTLE/REST data can never be
    labelled with the trial's target (REST is not 'low'). None for aborted,
    unsaved, quality-flagged or (by default) not-accepted attempts. Selecting
    samples by host_mono still needs margins: batch receipt time is not the
    acquisition time of each sample.
    """
    cand = rec.get("training_candidate") or {}
    if cand.get("phase") != "HOLD" or cand.get("quality_status") != "clean":
        return None
    cat = attempt_category(rec)
    if cat == "invalid" or (require_accepted and cat != "accepted"):
        return None
    label = rec["label"]
    return {"start_host_mono": cand["start_host_mono"], "end_host_mono": cand["end_host_mono"],
            "target_label": label["target_label"], "target_class_id": label["target_class_id"],
            "target_s": label["target_s"], "label_source": label["label_source"],
            "motion_condition": motion_of(rec)}


# -- object catalog ---------------------------------------------------------------

def merge_objects(saved: Optional[Iterable]) -> List[Tuple[str, str]]:
    """Default objects (stable IDs, renamable) plus valid extra objects."""
    names = dict(DEFAULT_OBJECTS)
    extra = []
    for item in saved or ():
        try:
            oid, name = str(item[0]), str(item[1]).strip()
        except (TypeError, IndexError):
            continue
        if not valid_key(oid):
            continue
        if oid in names:
            names[oid] = name or names[oid]
        elif oid not in dict(extra):
            extra.append((oid, name or oid))
    return [(oid, names[oid]) for oid, _ in DEFAULT_OBJECTS] + extra


def load_objects(output_root: Path) -> List[Tuple[str, str]]:
    p = Path(output_root) / OBJECTS_FILE
    try:
        return merge_objects(read_json(p).get("objects"))
    except FileNotFoundError:
        return merge_objects(None)


def objects_doc(objects: List[Tuple[str, str]]) -> dict:
    return {"schema": "gforce_emg_ui.stiffness_objects", "schema_version": 1,
            "note": "Object IDs are stable across sessions; names are descriptive only.",
            "objects": [[oid, name] for oid, name in objects]}


def target_counts(plan_counts: Dict[str, Tuple[int, int, int]]) -> Dict[str, int]:
    return {t: sum(c[i] for c in plan_counts.values()) for i, t in enumerate(TARGET_ORDER)}
