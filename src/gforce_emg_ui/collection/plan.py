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

The next trial is drawn at random (pick_trial). One-trial edits of its object,
target or movement are resolved to an existing incomplete planned trial with
exactly those values (resolve_request); the plan itself is never changed.

Surface press (task condition, not an object) is added to an existing standard
200-trial v2 plan only by an explicit, one-time extension (extend_plan_file):
30 stationary trials appended (10 per target), same plan_id, original trials
unchanged, backup kept, file replaced atomically. Extended plans are schema 3,
so older app versions refuse them instead of instructing a grasp. Surface press
is opt-in for the random selection (ticked like an object).
"""

import collections
import copy
import hashlib
import json
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .protocol import (GRASP, MOTION_INSTRUCTION_VERSION, MOTION_ORDER, MOTIONS, STATIONARY,
                       SURFACE_PRESS, SURFACE_PRESS_NAME, SURFACE_PRESS_POSTURE, TARGET_ORDER,
                       TASK_INSTRUCTION_VERSIONS)
from .storage import atomic_write_text, dumps, read_json, valid_key

EMPTY_HAND = "empty_hand"
DEFAULT_OBJECTS = (
    (EMPTY_HAND, "Empty hand"),
    ("obj01", "Object 1"), ("obj02", "Object 2"), ("obj03", "Object 3"),
    ("obj04", "Object 4"), ("obj05", "Object 5"),
)
DEFAULT_POSTURES = {EMPTY_HAND: ("fist", "Closed fist, no object"),
                    SURFACE_PRESS: (SURFACE_PRESS_POSTURE, "Index fingertip pressed on the desk")}
# Task conditions: not objects, outside the movement split, stationary only, and
# eligible for the random pick only when ticked (opt-in)
TASK_CONDITIONS = {SURFACE_PRESS: SURFACE_PRESS_NAME}
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
STANDARD_TOTAL = 200

SURFACE_PRESS_EXTENSION = "surface_press/v1"
SURFACE_PRESS_COUNTS = (10, 10, 10)
EXTENSION_GENERATOR = "weighted-random-order/no-target-run-over-2/surface-press/v1"
PLAN_BACKUP_DIR = "plan_backups"          # not plan_*.json: never loaded as a plan
EXTENDABLE, EXTENDED, INCOMPATIBLE = "extendable", "extended", "incompatible"


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
    extensions: List[dict] = field(default_factory=list)   # v3: appended task trials

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
        return 1 if self.motions is None else 3 if self.extensions else 2

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
            split = motion_split({k: v for k, v in self.counts.items()
                                  if k not in TASK_CONDITIONS}, self.motions)
            for obj in set(self.counts) & set(TASK_CONDITIONS):     # stationary only
                for i, t in enumerate(TARGET_ORDER):
                    for m in self.motions:
                        split[(obj, t, m)] = self.counts[obj][i] if m == STATIONARY else 0
            d["motion_conditions"] = list(self.motions)
            d["motion_instruction_version"] = MOTION_INSTRUCTION_VERSION
            d["motion_counts"] = {obj: {t: {m: split[(obj, t, m)] for m in self.motions}
                                        for t in TARGET_ORDER} for obj in self.counts}
        if self.extensions:
            d["extensions"] = self.extensions
        d.update(total=self.total, trials=self.trials)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TrialPlan":
        if d.get("schema") != PLAN_SCHEMA:
            raise ValueError("not a stiffness plan file")
        version = d.get("schema_version", 1)
        if version not in (1, 2, 3):
            raise ValueError(f"unsupported plan schema_version {version}")
        motions, extensions = None, []
        if version >= 2:
            motions = tuple(d["motion_conditions"])
            unknown = {t.get("motion_condition") for t in d["trials"]} - set(MOTIONS)
            if set(motions) - set(MOTIONS) or unknown:
                raise ValueError(f"unknown motion condition(s) in plan: {sorted(map(str, unknown))}")
        if version == 3:
            extensions = list(d.get("extensions") or [])
            if not extensions:
                raise ValueError("plan schema_version 3 without extensions")
            if any(t["object_id"] in TASK_CONDITIONS and t.get("motion_condition") != STATIONARY
                   for t in d["trials"]):
                raise ValueError("surface press trials must be stationary")
        return cls(plan_id=d["plan_id"], participant_id=d["participant_id"],
                   data_source=d["data_source"], seed=int(d["seed"]), created=d["created"],
                   counts={k: tuple(v) for k, v in d["counts_low_medium_high"].items()},
                   trials=list(d["trials"]), motions=motions, extensions=extensions)


# -- surface-press extension of an existing plan ------------------------------------

def surface_press_status(plan: TrialPlan) -> Tuple[str, str]:
    """(extendable | extended | incompatible, explanation). Only an unmodified standard
    200-trial v2 plan (default counts, the four movements) may be extended."""
    if any(e.get("id") == SURFACE_PRESS_EXTENSION for e in plan.extensions):
        return EXTENDED, (f"Plan {plan.plan_id} already has its 30 Surface Press trials "
                          f"({plan.total} planned trials). Nothing was added.")
    why = []
    if plan.motions is None:
        why.append("plan v1 (no movement assigned)")
    elif tuple(plan.motions) != PLAN_MOTIONS:
        why.append(f"movements {list(plan.motions)} are not the standard four")
    if plan.extensions:
        why.append("plan already has other extensions")
    if plan.counts != {k: tuple(v) for k, v in DEFAULT_PLAN_COUNTS.items()}:
        why.append("condition counts are not the standard 200-trial counts")
    if plan.total != STANDARD_TOTAL:
        why.append(f"{plan.total} planned trials, not {STANDARD_TOTAL}")
    cells = collections.Counter((t["object_id"], t["target_label"]) for t in plan.trials)
    if cells != collections.Counter({(o, t): c[i] for o, c in plan.counts.items()
                                     for i, t in enumerate(TARGET_ORDER) if c[i]}):
        why.append("trials do not match the plan's condition counts")
    if len({t["trial_id"] for t in plan.trials}) != plan.total or \
            len({t["index"] for t in plan.trials}) != plan.total:
        why.append("duplicate trial IDs or indices")
    if why:
        return INCOMPATIBLE, (f"Plan {plan.plan_id} is not a standard 200-trial plan ("
                              + "; ".join(why) + "). Nothing was changed.")
    return EXTENDABLE, ""


def _free_trial_ids(plan: TrialPlan, n: int) -> List[Tuple[int, str]]:
    """(index, trial_id) pairs after the last index; IDs already used are skipped."""
    used = {t["trial_id"] for t in plan.trials}
    i = max((t["index"] for t in plan.trials), default=0)
    k, out = i, []
    while len(out) < n:
        k += 1
        if f"t{k:03d}" not in used:
            out.append((i + len(out) + 1, f"t{k:03d}"))
    return out


def extend_with_surface_press(plan: TrialPlan, seed: Optional[int] = None,
                              added: Optional[str] = None, **record) -> TrialPlan:
    """New TrialPlan = plan + 30 stationary surface-press trials (pure; plan unchanged)."""
    status, why = surface_press_status(plan)
    if status != EXTENDABLE:
        raise ValueError(why)
    if seed is None:                         # reproducible from the plan seed, and saved
        seed = random.Random(f"{plan.seed}:{SURFACE_PRESS_EXTENSION}").randrange(1, 2 ** 31)
    cells = {(SURFACE_PRESS, t, STATIONARY): n for t, n in zip(TARGET_ORDER, SURFACE_PRESS_COUNTS)}
    order = _weighted_order(cells, int(seed), MAX_TARGET_RUN)
    new = [{"index": i, "trial_id": tid, "object_id": SURFACE_PRESS,
            "posture_id": SURFACE_PRESS_POSTURE, "target_label": tgt,
            "motion_condition": STATIONARY, "trial_type": SURFACE_PRESS}
           for (i, tid), (_, tgt, _) in zip(_free_trial_ids(plan, len(order)), order)]
    ext = {"id": SURFACE_PRESS_EXTENSION, "condition": SURFACE_PRESS, "display": SURFACE_PRESS_NAME,
           "trial_type": SURFACE_PRESS, "posture_id": SURFACE_PRESS_POSTURE,
           "motion_condition": STATIONARY,
           "task_instruction_version": TASK_INSTRUCTION_VERSIONS[SURFACE_PRESS],
           "counts_low_medium_high": list(SURFACE_PRESS_COUNTS), "seed": int(seed),
           "generator": EXTENSION_GENERATOR, "max_target_run": MAX_TARGET_RUN,
           "added": added or datetime.now().isoformat(timespec="seconds"),
           "total_before": plan.total, "total_after": plan.total + len(new),
           "added_trial_ids": [t["trial_id"] for t in new], **record}
    return TrialPlan(plan_id=plan.plan_id, participant_id=plan.participant_id,
                     data_source=plan.data_source, seed=plan.seed, created=plan.created,
                     counts={**plan.counts, SURFACE_PRESS: SURFACE_PRESS_COUNTS},
                     trials=copy.deepcopy(plan.trials) + new, motions=plan.motions,
                     extensions=copy.deepcopy(plan.extensions) + [ext])


def extension_problems(original: TrialPlan, extended: TrialPlan) -> List[str]:
    """Checks of a complete extended plan before it is written (empty = valid)."""
    p, n = [], original.total
    for k in ("plan_id", "participant_id", "data_source", "seed", "created", "motions"):
        if getattr(original, k) != getattr(extended, k):
            p.append(f"{k} changed")
    if [dumps(t) for t in extended.trials[:n]] != [dumps(t) for t in original.trials]:
        p.append("original trials changed")
    added = extended.trials[n:]
    if len(added) != sum(SURFACE_PRESS_COUNTS):
        p.append(f"{len(added)} trials added, not {sum(SURFACE_PRESS_COUNTS)}")
    if collections.Counter(t["target_label"] for t in added) != \
            dict(zip(TARGET_ORDER, SURFACE_PRESS_COUNTS)):
        p.append("added trials are not 10 low / 10 medium / 10 high")
    if any((t["object_id"], t["posture_id"], t["motion_condition"], t.get("trial_type")) !=
           (SURFACE_PRESS, SURFACE_PRESS_POSTURE, STATIONARY, SURFACE_PRESS) for t in added):
        p.append("added trials are not stationary surface press")
    if len({t["trial_id"] for t in extended.trials}) != extended.total or \
            len({t["index"] for t in extended.trials}) != extended.total:
        p.append("duplicate trial IDs or indices")
    cells = collections.Counter((t["object_id"], t["target_label"]) for t in extended.trials)
    if any(cells[(o, t)] != c[i] for o, c in extended.counts.items()
           for i, t in enumerate(TARGET_ORDER)):
        p.append("condition counts do not match the trials")
    d = extended.to_dict()
    if d["total"] != n + len(added) or TrialPlan.from_dict(json.loads(dumps(d))).to_dict() != \
            json.loads(dumps(d)):
        p.append("extended plan does not round-trip")
    return p


def extend_plan_file(path: Path, expected: TrialPlan, seed: Optional[int] = None) -> TrialPlan:
    """Extend the saved plan file in place (I/O thread). Transactional: the file must
    still equal the loaded plan; the complete new plan is validated, the original is
    copied to plan_backups/ (verified), then replaced atomically and read back. Any
    failure before the replace leaves the original file untouched; a bad read-back
    restores it. Raises ValueError (refused) or OSError (I/O)."""
    path = Path(path)
    raw = path.read_bytes()
    if json.loads(raw) != json.loads(dumps(expected.to_dict())):
        raise ValueError(f"{path.name} differs from the loaded plan: reload the participant "
                         "and try again. Nothing was changed.")
    original = TrialPlan.from_dict(json.loads(raw))
    status, why = surface_press_status(original)
    if status != EXTENDABLE:
        raise ValueError(why)
    backups = path.parent / PLAN_BACKUP_DIR
    backup = backups / f"{path.stem}.before-surface-press.json"
    if backup.exists() and backup.read_bytes() != raw:
        backup = backups / f"{path.stem}.before-surface-press.{uuid.uuid4().hex[:8]}.json"
    sha = hashlib.sha256(raw).hexdigest()
    extended = extend_with_surface_press(
        original, seed, original_sha256=sha,
        backup=f"{PLAN_BACKUP_DIR}/{backup.name}")
    problems = extension_problems(original, extended)
    if problems:
        raise ValueError("Extended plan failed validation (" + "; ".join(problems) +
                         "). Nothing was changed.")
    text = dumps(extended.to_dict())
    backups.mkdir(exist_ok=True)
    if not backup.exists():
        atomic_write_text(backup, raw.decode())
    if hashlib.sha256(backup.read_bytes()).hexdigest() != sha:
        raise OSError(f"backup {backup} could not be verified; plan not changed")
    atomic_write_text(path, text)
    if path.read_bytes() != text.encode():
        atomic_write_text(path, raw.decode())
        raise OSError(f"{path.name} read back differently; original restored")
    return extended


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
            and cond.get("trial_type", GRASP) == planned.get("trial_type", GRASP)
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
    by_motion: Dict[str, List[int]] = field(default_factory=dict)            # v2+, grasp only
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
        if "motion_condition" in t and t["object_id"] not in TASK_CONDITIONS:
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
                if "motion_condition" in planned and planned["object_id"] not in TASK_CONDITIONS:
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


SELECTION_RULE = "random-available/no-target-run-over-2/v1"


def eligible_trials(plan: TrialPlan, progress: Progress, available: Iterable[str]) -> List[int]:
    """Indices of incomplete planned trials whose object is available. Empty hand
    needs no object: it is always eligible until its planned trials are complete."""
    avail = set(available) | {EMPTY_HAND}
    return [i for i, t in enumerate(plan.trials)
            if t["trial_id"] not in progress.completed and t["object_id"] in avail]


def run_banned_target(recent_targets: Iterable[str], max_run: int = MAX_TARGET_RUN) -> Optional[str]:
    """Target that would make a run of more than max_run, or None."""
    tail = list(recent_targets)[-max_run:]
    return tail[0] if len(tail) == max_run and len(set(tail)) == 1 else None


def pick_trial(plan: TrialPlan, progress: Progress, available: Iterable[str],
               recent_targets: Iterable[str] = (), key: str = "",
               exclude: Optional[int] = None, max_run: int = MAX_TARGET_RUN) -> Optional[int]:
    """Random next planned trial (index) among the eligible ones, or None.

    Deterministic for (plan seed, key, eligible set): re-ticking the same objects
    gives the same pick, so toggling cannot re-roll it. Avoids more than max_run
    identical targets in a row (recent_targets = last attempts) and the excluded
    index (a just-rejected trial) whenever another choice exists.
    """
    idx = eligible_trials(plan, progress, available)
    if exclude is not None and len(idx) > 1:
        idx = [i for i in idx if i != exclude]
    if not idx:
        return None
    banned = run_banned_target(recent_targets, max_run)
    pool = [i for i in idx if plan.trials[i]["target_label"] != banned] or idx
    return random.Random(f"{plan.seed}:{key}").choice(pool)


# -- one-trial edits of the next planned trial ---------------------------------------

SELECTION_FIELDS = ("object_id", "target_label", "motion_condition")
FIELD_NAMES = {"object_id": "object", "target_label": "target", "motion_condition": "movement"}
MANUAL_SELECTION_RULE = "manual-override/exact-match/proposal-kept-or-lowest-index/v1"


def trial_values(plan: TrialPlan, trial: dict) -> Dict[str, str]:
    """Object / target / movement a planned trial is recorded with (v1 plans: stationary)."""
    return {"object_id": trial["object_id"], "target_label": trial["target_label"],
            "motion_condition": plan.motion_of(trial)}


def describe(values: dict) -> str:
    """'obj02 / medium / reach/sweep'."""
    m = values.get("motion_condition")
    return " / ".join([values.get("object_id") or "?", values.get("target_label") or "?",
                       MOTIONS[m]["display"].lower() if m in MOTIONS else (m or "?")])


def resolve_request(plan: TrialPlan, progress: Progress, request: dict,
                    prefer: Iterable[Tuple[str, Optional[int]]] = ()
                    ) -> Tuple[Optional[int], str, str]:
    """(index, how, problem) of an incomplete planned trial with exactly the requested
    object, target and movement.

    prefer: (how, index) pairs tried first (the original proposal, the current
    resolution); otherwise the lowest plan index. Requested values are never
    changed and plan entries are never modified, so a request without a remaining
    match stays as it is with a problem text.
    """
    missing = [FIELD_NAMES[f] for f in SELECTION_FIELDS if not request.get(f)]
    if missing:
        return None, "", f"Choose {' and '.join(missing)}."
    if request["object_id"] == SURFACE_PRESS and SURFACE_PRESS not in plan.counts:
        return None, "", ("This plan has no Surface Press trials: add them under Protocol & "
                          "plan settings, choose another object, or reset edits.")
    if request["object_id"] not in plan.counts:
        return None, "", (f"{request['object_id']} has no trials in this plan. Choose another "
                          "object, reset edits, or use Manual mode for off-plan objects.")
    if request["object_id"] in TASK_CONDITIONS and request["motion_condition"] != STATIONARY:
        return None, "", ("Surface Press is stationary only: choose Stationary, another "
                          "object, or reset edits.")
    if plan.motions is None and request["motion_condition"] != V1_PLAN_MOTION:
        return None, "", "Plan v1 has no movement: choose Stationary or reset edits."
    want = {f: request[f] for f in SELECTION_FIELDS}
    idx = [i for i, t in enumerate(plan.trials)
           if t["trial_id"] not in progress.completed and trial_values(plan, t) == want]
    if not idx:
        return None, "", (f"No remaining planned trials for {describe(want)}. "
                          "Change a setting or reset edits.")
    for how, i in prefer:
        if i in idx:
            return i, how, ""
    return idx[0], "lowest_index", ""


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
        if not valid_key(oid) or oid in TASK_CONDITIONS:   # conditions are not objects
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
