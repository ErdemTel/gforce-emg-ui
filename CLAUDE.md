# CLAUDE.md

Project context, goals and progress live in `memory.md`:

@memory.md

## Working rules

- At the **end of every session**, append a dated entry (YYYY-MM-DD) to the *Progress Log* in `memory.md`
  summarizing what was achieved, decisions made, and next steps. Update *Open decisions* when one is resolved.
- Use the project venv: `.venv/bin/python` (create it per README if missing).
- This is a shared Linux account: never modify global git/gh config; Erdem's gh credentials live in `~/erdem/.config/gh` (repo-local credential helper).
- Priority for any EMG code: minimal latency from BLE notification to display/disk.
