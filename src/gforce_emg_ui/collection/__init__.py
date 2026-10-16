"""Guided, timed stiffness-command data collection (Phase 0: data collection only).

protocol    target labels, instructions, timing settings, trial state machine (Qt-free)
readiness   stream / electrode-contact readiness and per-trial quality capture
plan        200-trial development plan, attempt index and progress from trial.json files
storage     atomic metadata writes on a background thread, folder names, git info
controller  one attempt at a time: readiness -> Recorder -> phases -> trial.json -> review
"""
