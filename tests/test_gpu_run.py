"""Tests for bin/gpu_run scheduler lifecycle wrapper."""
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
GPU_RUN = REPO_ROOT / "bin" / "gpu_run"


def _make_fake_scheduler(tmp_path: Path, poll_sequence: list[str]) -> tuple[Path, Path]:
    state_file = tmp_path / "sched_log.json"
    state_file.write_text(json.dumps({"calls": [], "polls": list(poll_sequence)}))
    sched = tmp_path / "fake_scheduler.py"
    sched.write_text(
        f"""#!/usr/bin/env python3
import json, sys
from pathlib import Path

state_path = Path({str(state_file)!r})
state = json.loads(state_path.read_text())
args = sys.argv[1:]
state["calls"].append(args)
cmd = args[0] if args else ""

if cmd == "poll":
    polls = state["polls"]
    out = polls.pop(0) if polls else "claimed"
    state_path.write_text(json.dumps(state))
    print(out)
elif cmd == "activate":
    state_path.write_text(json.dumps(state))
    print("activated")
elif cmd == "heartbeat":
    state_path.write_text(json.dumps(state))
    print("ok")
elif cmd == "release":
    state_path.write_text(json.dumps(state))
    print("released")
else:
    state_path.write_text(json.dumps(state))
    print("job-123")
"""
    )
    sched.chmod(0o755)
    return sched, state_file


def test_gpu_run_polls_activates_runs_heartbeats_and_releases_completed(tmp_path: Path):
    sched, state_file = _make_fake_scheduler(tmp_path, ["not_my_turn", "claimed"])
    proc = subprocess.run(
        [
            sys.executable,
            str(GPU_RUN),
            "--gpu", "4090",
            "--job-id", "job-abc",
            "--scheduler", str(sched),
            "--poll-interval", "0.05",
            "--heartbeat-interval", "0.05",
            "--vram-used", "16",
            "--",
            sys.executable, "-c", "import time; time.sleep(0.12)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.returncode == 0
    calls = json.loads(state_file.read_text())["calls"]
    verbs = [c[0] for c in calls]
    assert verbs[:3] == ["poll", "poll", "activate"]
    assert "heartbeat" in verbs
    assert verbs[-1] == "release"
    assert "--status" in calls[-1] and "completed" in calls[-1]


def test_gpu_run_releases_failed_when_command_exits_nonzero(tmp_path: Path):
    sched, state_file = _make_fake_scheduler(tmp_path, ["claimed"])
    proc = subprocess.run(
        [
            sys.executable,
            str(GPU_RUN),
            "--gpu", "4090",
            "--job-id", "job-fail",
            "--scheduler", str(sched),
            "--poll-interval", "0.05",
            "--heartbeat-interval", "0.5",
            "--",
            sys.executable, "-c", "import sys; sys.exit(7)",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 7
    calls = json.loads(state_file.read_text())["calls"]
    assert calls[-1][0] == "release"
    assert "failed" in calls[-1]


def test_gpu_run_exits_when_job_not_queued(tmp_path: Path):
    sched, _ = _make_fake_scheduler(tmp_path, ["not_queued"])
    proc = subprocess.run(
        [
            sys.executable,
            str(GPU_RUN),
            "--gpu", "4090",
            "--job-id", "job-gone",
            "--scheduler", str(sched),
            "--",
            sys.executable, "-c", "pass",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2
