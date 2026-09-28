#!/usr/bin/env python3
"""Retire dead-owner queue entries from the file-based GPU scheduler.

WHY THIS EXISTS
    poll() enforces strict FIFO on `requested_at` (gpu_scheduler.py:825-827):
    any *queued* entry with an earlier timestamp makes every job behind it
    answer `not_my_turn`. read_queue() only surfaces status == "queued"
    (gpu_scheduler.py:488), so a queued entry whose requester has died blocks
    the entire queue — the card idles while every job behind it is refused.

    The scheduler's own reaper, expire_stale_queue_entries(max_age_h), is the
    sanctioned fix but defaults to a 24 h age and the CLI (`expire --help`)
    exposes no override. This script calls the same module function with a
    shorter age.

WHAT IT DOES
    Calls gpu_scheduler.expire_stale_queue_entries(max_age_h=AGE) — the module's
    own code path: atomic write (write_json_atomic) plus an events.log record.
    It flips status "queued" -> "expired" and never deletes the file, so the
    entry survives as an audit record.

USAGE
    python scratch/expire_stale_queue_entries.py [AGE_HOURS]     # default 2.0
"""
import sys

sys.path.insert(0, "/mnt/nas-ai-models/gpu-scheduler")
import gpu_scheduler as gs  # noqa: E402

age = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0

print("queue BEFORE:")
for j in gs.read_queue("4090"):
    print(f"  {j['id']} | {j['project']} | {j['vram_gb']}GB "
          f"| {j['requested_at']} | status={j['status']}")

expired = gs.expire_stale_queue_entries(max_age_h=age)
print(f"\nexpired (age > {age}h): {expired if expired else 'none'}")

print("queue AFTER:")
for j in gs.read_queue("4090"):
    print(f"  {j['id']} | {j['project']} | {j['vram_gb']}GB "
          f"| {j['requested_at']} | status={j['status']}")