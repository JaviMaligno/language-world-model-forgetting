from __future__ import annotations
import shlex
import subprocess

def run_in_sandbox(commands: list[str], scratch_dir: str,
                   timeout: float = 10.0) -> list[tuple[str, str]]:
    """Run each command sequentially in a bash session rooted at scratch_dir.

    Returns (command, observation) pairs. Observation = stdout+stderr plus an
    exit-code marker. Each command runs as an independent bash -c subprocess
    rooted at scratch_dir; only FILESYSTEM state persists across commands (via
    the shared dir), NOT shell-variable/env state.
    """
    out: list[tuple[str, str]] = []
    for cmd in commands:
        wrapped = f"cd {shlex.quote(scratch_dir)} && {cmd}"
        proc = subprocess.Popen(
            ["bash", "-c", wrapped],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            obs = stdout + stderr
            obs += f"\n[exit {proc.returncode}]"
        except subprocess.TimeoutExpired as exc:
            # exc.process may be None in older stdlib; fall back to proc
            child = getattr(exc, "process", None)
            if child is None:  # be defensive; .process may be None in older stdlib
                child = proc
            try:
                child.kill()
                child.communicate()
            except Exception:
                pass
            obs = f"[timeout after {timeout}s]"
        out.append((cmd, obs.strip()))
    return out
