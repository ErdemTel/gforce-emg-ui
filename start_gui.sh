#!/usr/bin/env bash
# Open the existing unified EMG/manual-control GUI using its existing venv.
# No ROS sourcing, dependency installation, or shell-profile changes.
set -eo pipefail
gui_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -x "$gui_dir/.venv/bin/python" ]]; then
    echo "Existing GUI environment not found: $gui_dir/.venv/bin/python" >&2
    exit 2
fi
cd -- "$gui_dir"
exec env -u PYTHONPATH -u LD_LIBRARY_PATH \
    "$gui_dir/.venv/bin/python" -m gforce_emg_ui "$@"
