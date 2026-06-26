from __future__ import annotations
import subprocess

def run_in_sandbox(commands: list[str], scratch_dir: str,
                   timeout: float = 10.0) -> list[tuple[str, str]]:
    """Run each command sequentially in a bash session rooted at scratch_dir.

    Returns (command, observation) pairs. Observation = stdout+stderr plus an
    exit-code marker. State (cwd, files, env) persists across commands within
    one call because they run in a single bash -c chained session per command
    but share scratch_dir; for cross-command shell state we serialize each
    command and re-cd, keeping filesystem state (sufficient for our scenarios).
    """
    out: list[tuple[str, str]] = []
    for cmd in commands:
        wrapped = f"cd {scratch_dir!r} && {cmd}"
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
