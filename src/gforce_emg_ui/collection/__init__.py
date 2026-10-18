"""Guided, timed stiffness-command data collection (Phase 0: data collection only).

protocol    target labels, instructions, timing settings, trial state machine (Qt-free)
readiness   stream / electrode-contact readiness and per-trial quality capture
plan        development plan (200 trials, +30 surface press), attempts, progress
storage     atomic metadata writes on a background thread, folder names, git info
controller  one attempt at a time: readiness -> Recorder -> phases -> trial.json -> review
"""
