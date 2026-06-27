from __future__ import annotations
import shlex
import subprocess

def run_in_sandbox(commands: list[str], scratch_dir: str,
                   timeout: float = 10.0) -> list[tuple[str, str]]:
    """Run each command as an independent `bash -c` subprocess rooted at
    scratch_dir, sequentially. Returns (command, observation) pairs where
    observation = stdout+stderr plus an exit-code marker (or a timeout marker).
    Only FILESYSTEM state persists across commands (they share scratch_dir);
    shell-variable/env state does NOT persist.
    """
    out: list[tuple[str, str]] = []
    for cmd in commands:
        wrapped = f"cd {shlex.quote(scratch_dir)} && {cmd}"
        try:
            proc = subprocess.run(
                ["bash", "-c", wrapped],
                capture_output=True, text=True, timeout=timeout,
            )
            obs = proc.stdout + proc.stderr
            obs += f"\n[exit {proc.returncode}]"
        except subprocess.TimeoutExpired:
            obs = f"[timeout after {timeout}s]"
        out.append((cmd, obs.strip()))
    return out
