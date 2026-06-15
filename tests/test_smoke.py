"""Non-GPU smoke tests for VL-GRiP3.

These exercise the pure-Python logic that does not need a GPU, the heavy ML
stack, or robot hardware:

  * command-prompt parsing (main.process_user_prompt)
  * the robot safety layer (pose validation, speed clamp, dry-run) in
    UR3Commands, by stubbing the ur_rtde binary dependency.

Run with:  python tests/test_smoke.py
(Exit code 0 = all passed.)
"""
import os
import sys
import types

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

# --- Stub the ur_rtde binary modules so class_robot_library imports cleanly ---
for _name in ("rtde_control", "rtde_receive"):
    _m = types.ModuleType(_name)
    _m.RTDEControlInterface = object
    _m.RTDEReceiveInterface = object
    sys.modules[_name] = _m

import main  # noqa: E402
from GRiP3_Pipeline.utils.class_robot_library import UR3Commands  # noqa: E402


def test_process_user_prompt():
    seg, target, cmd = main.process_user_prompt("move tac2 in red target")
    assert seg == "<image> segment tac2", seg
    assert target == "<image> detect red target", target
    assert cmd == "<image> move in red target", cmd

    seg, target, cmd = main.process_user_prompt("move the black cube positive x by 0.05")
    assert seg == "<image> segment black cube", seg
    assert target is None, target
    assert cmd == "<image> shift positive x by 0.05", cmd

    raised = False
    try:
        main.process_user_prompt("hello there")
    except ValueError:
        raised = True
    assert raised, "expected ValueError on an unparsable prompt"


def test_pose_validation():
    ur = UR3Commands(dry_run=True)
    assert ur._validate_pose([0.2, -0.2, 0.3, 1.0, -2.0, 0.0]) is True
    assert ur._validate_pose([0.2, -0.2, 0.3]) is False          # wrong length
    assert ur._validate_pose([float("nan"), 0, 0, 0, 0, 0]) is False  # NaN
    assert ur._validate_pose([99.0, 0, 0, 0, 0, 0]) is False      # outside workspace


def test_speed_clamped():
    ur = UR3Commands(dry_run=True, speed=10.0, acceleration=10.0)
    assert ur.speed <= 0.25, ur.speed
    assert ur.acceleration <= 0.5, ur.acceleration


def test_dry_run_execution():
    """A full command sequence must run without hardware and never raise."""
    ur = UR3Commands(dry_run=True)
    ur.connect()  # dry-run: must not raise even with no robot
    assert ur.get_current_tcp_pose() is not None  # returns home pose in dry-run
    command_line = ("connect ; approach ; gripper open ; grasp ; gripper close ; "
                    "positive(z,0.05) ; home pose ; disconnect")
    main.execute_command_sequence(ur, command_line)  # must not raise


def main_():
    tests = [
        test_process_user_prompt,
        test_pose_validation,
        test_speed_clamped,
        test_dry_run_execution,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main_())
