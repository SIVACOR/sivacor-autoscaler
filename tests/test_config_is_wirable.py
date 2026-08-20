"""Every environment variable this controller reads must be declared in the stack file.

Docker Swarm passes a service **only** the variables its `environment:` block names.
Setting one in `.env` that the stack file does not mention has no effect at all -- so a
knob wired into `build_config()` and documented in ENVIRONMENT.md can still be
completely inert, and the symptom is that turning it on changes nothing.

That is exactly what happened to `SIVACOR_MAX_VCPUS` and `SIVACOR_MAX_RAM_GB` on
2026-08-20: read by `__main__.py`, documented in ENVIRONMENT.md, absent from
`docker-stack.autoscaler.yml`. Caught by review, not by anything automatic, and the next
step would have been a quota test that "passed" because the check never ran.

It is the same failure family as D9's readiness marker and D8 Option C shipping inert for
two loop-test runs: the code is right, the wiring is missing, and nothing fails loudly.
This test is the cheap way to stop repeating it.

Skipped when `deploy-sivacor` is not checked out beside this repo, matching
`test_fleet.py`'s treatment of the cloud-init template.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MAIN = Path(__file__).resolve().parents[1] / "sivacor_autoscaler" / "__main__.py"
STACK = ROOT / "deploy-sivacor" / "docker-stack.autoscaler.yml"

#: Read by this process but deliberately NOT supplied by the autoscaler stack file.
#: Each needs a reason, because "it is missing" and "it is missing on purpose" look
#: identical from here and only one of them is a bug.
NOT_FROM_THE_STACK_FILE = {
    # Girder's own variables, set on the shared girder-sivacor service definition
    # rather than per-service -- the controller runs from that image (P0.5).
    "GIRDER_MONGO_URI",
    "GIRDER_WORKER_BROKER",
    "MASTER_KEY_HEX",
    "REDIS_PASSWORD",
}


def _env_names(source: str) -> set[str]:
    """Every name passed to `_env(...)` or looked up in `os.environ`."""
    names = set(re.findall(r'_env\(\s*"([A-Z0-9_]+)"', source))
    names |= set(re.findall(r'os\.environ\.get\(\s*"([A-Z0-9_]+)"', source))
    return names


@pytest.mark.skipif(not STACK.is_file(), reason="deploy-sivacor not checked out")
def test_every_variable_the_controller_reads_is_declared_in_the_stack_file():
    read = _env_names(MAIN.read_text())
    assert read, "no variables found -- has _env() been renamed?"

    declared = set(re.findall(r"^\s*-\s*([A-Z0-9_]+)=", STACK.read_text(), re.MULTILINE))

    missing = {n for n in read if n not in declared} - NOT_FROM_THE_STACK_FILE
    assert not missing, (
        "read by __main__.py but not declared in docker-stack.autoscaler.yml, so "
        "setting them in .env does nothing: " + ", ".join(sorted(missing)) + ". Add "
        "them to the autoscaler service's `environment:` block, or list them in "
        "NOT_FROM_THE_STACK_FILE with the reason."
    )


@pytest.mark.skipif(not STACK.is_file(), reason="deploy-sivacor not checked out")
def test_the_stack_file_does_not_declare_variables_nothing_reads():
    """The other direction, which is confusing rather than broken.

    A declared variable nothing reads is a knob an operator can set, redeploy for, and
    watch do nothing -- the same lived experience as the bug above, from the opposite
    cause. Only SIVACOR_* is checked: the service legitimately carries Girder's and the
    broker's variables for the image it shares.
    """
    declared = {
        n
        for n in re.findall(r"^\s*-\s*([A-Z0-9_]+)=", STACK.read_text(), re.MULTILINE)
        if n.startswith("SIVACOR_")
    }
    read = _env_names(MAIN.read_text())
    # SIVACOR_AUTOSCALING gates whether the stack file is included at all, and
    # SIVACOR_DIAGNOSTICS_HOSTPATH is a bind-mount path, not a process variable.
    known_non_process = {"SIVACOR_AUTOSCALING", "SIVACOR_DIAGNOSTICS_HOSTPATH"}
    unread = declared - read - known_non_process
    assert not unread, (
        "declared in docker-stack.autoscaler.yml but read by nothing in __main__.py: "
        + ", ".join(sorted(unread))
        + ". Either wire it up or delete it -- a knob that does nothing is worse than "
        "an absent one, because an operator will set it and believe it took."
    )
