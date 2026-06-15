import json
import time
import math
import logging

from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

from .robotiq_gripper_control import RobotiqGripper
from .paths import SCENE_DIR, ROBOT_IP

logger = logging.getLogger(__name__)

# Conservative default Cartesian workspace (meters, robot base frame). This is a
# safety guard against grossly invalid target poses, NOT a calibrated cell
# boundary. Re-measure / tighten it for your own cell (e.g. a UR5e).
DEFAULT_WORKSPACE_BOUNDS = {
    "x": (-0.6, 0.6),
    "y": (-0.6, 0.6),
    "z": (-0.05, 0.6),
}
# Hard caps on Cartesian speed / acceleration to avoid dangerous fast motions.
MAX_SPEED = 0.25          # m/s
MAX_ACCELERATION = 0.5    # m/s^2


class UR3Commands:
    """
    UR robot command library via RTDE.

    The class name is kept for backwards compatibility, but it works with any
    UR arm: set ``robot_ip``, the target/home poses and ``workspace_bounds`` for
    your own cell (e.g. a UR5e — the bundled poses were calibrated for the
    original UR3 cell and must be re-measured).

    Safety features:
        - Every Cartesian target is checked for NaN/inf and validated against a
          workspace envelope before any motion is commanded.
        - Speed / acceleration are clamped to MAX_SPEED / MAX_ACCELERATION.
        - ``dry_run=True`` logs intended motions without commanding the robot, so
          the full pipeline can be exercised end-to-end without hardware.

    Methods:
        - connect() / disconnect(): manage the RTDE connection.
        - move_to_pose(): validated Cartesian move.
        - move_to_grasping() / move_to_target() / move_to_home(): task moves.
        - gripper_open() / gripper_close(): gripper control.
        - approach(): de-approach move (z + 0.05 from the grasping pose).
        - get_current_tcp_pose(), positive_shift(), negative_shift().
    """

    def __init__(self, rtde_control_interface=None, rtde_receive_interface=None,
                 robot_ip=None,
                 dry_run=False,
                 move_delay=4.0,       # wait after a movement
                 gripper_delay=1.0,    # wait after a gripper operation
                 speed=0.08,
                 acceleration=0.1,
                 workspace_bounds=None):
        """
        :param robot_ip: UR controller IP (defaults to paths.ROBOT_IP / env).
        :param dry_run: if True, never command the robot; only log intended motions.
        :param speed/acceleration: Cartesian limits (clamped to safe maxima).
        :param workspace_bounds: dict {"x":(lo,hi),"y":(lo,hi),"z":(lo,hi)} in meters.
        """
        self.rtde_c = rtde_control_interface
        self.rtde_r = rtde_receive_interface
        self.robot_ip = robot_ip or ROBOT_IP
        self.dry_run = dry_run
        self.move_delay = move_delay
        self.gripper_delay = gripper_delay

        # Velocity and acceleration parameters for Cartesian motions (clamped).
        self.speed = min(speed, MAX_SPEED)
        self.acceleration = min(acceleration, MAX_ACCELERATION)

        self.workspace_bounds = workspace_bounds or DEFAULT_WORKSPACE_BOUNDS

        # Fixed poses for targets / home.
        # NOTE: calibrated for the original UR3 cell — re-measure for your cell.
        self._yellow_target_pose = [0.3506093066335152, 0.026065163598190763, 0.08630217204639501,
                                     1.85011627085003, -2.538905414868407, -3.633083091122777e-05]
        self._green_target_pose = [0.3092810912107801, -0.16723593843130086, 0.08633799189949548,
                                    1.0619274431058556, -2.956625921081819, 0.00014242509976082317]
        self._red_target_pose   = [-0.09621386458169016, -0.33818094057402187, 0.08633208721832436,
                                    -1.0218855726535594, -2.9706414975151993, 1.7414612154461717e-05]
        self._home_pose         = [0.19463757949913976, -0.23026603411194024, 0.27530714604309037,
                                    2.267162875376877, -2.174720802168727, 0.0001074923254140672]

        # Default key for grasping pose
        self.default_grasp_pose_key = "1"

        # Variable for gripper (initialized after connection)
        self.gripper = None

    # --------------------------- safety helpers ---------------------------
    def _validate_pose(self, pose):
        """Return True if ``pose`` is a finite 6-vector inside the workspace."""
        if pose is None or len(pose) != 6:
            logger.error("Invalid pose (expected 6 values): %r", pose)
            return False
        if any((v is None or not math.isfinite(v)) for v in pose):
            logger.error("Pose contains NaN/inf, refusing to move: %r", pose)
            return False
        x, y, z = pose[0], pose[1], pose[2]
        bx, by, bz = (self.workspace_bounds["x"],
                      self.workspace_bounds["y"],
                      self.workspace_bounds["z"])
        if not (bx[0] <= x <= bx[1] and by[0] <= y <= by[1] and bz[0] <= z <= bz[1]):
            logger.error("Target (x,y,z)=%s outside workspace bounds %s; refusing to move.",
                         [x, y, z], self.workspace_bounds)
            return False
        return True

    def _init_gripper(self):
        try:
            self.gripper = RobotiqGripper(self.rtde_c)
            logger.info("[UR3Commands] Activating the gripper...")
            self.gripper.activate()
            self.gripper.set_force(0)
            self.gripper.set_speed(30)
        except Exception as e:
            logger.error("[UR3Commands] Error initializing gripper: %s", e)
            self.gripper = None

    def connect(self):
        """
        Establish the RTDE connection. Raises on failure so the caller can abort
        cleanly. In dry-run mode no real connection is opened.
        """
        robot_ip = self.robot_ip
        if self.dry_run:
            logger.info("[connect] dry-run: skipping real RTDE connection to %s.", robot_ip)
            self.gripper = None
            return
        try:
            self.rtde_c = RTDEControlInterface(robot_ip)
            self.rtde_r = RTDEReceiveInterface(robot_ip)
            if self.rtde_c is None or self.rtde_r is None:
                raise RuntimeError(f"RTDE interfaces not initialized for {robot_ip}")
            logger.info("[connect] Connection established with the robot %s.", robot_ip)
            self._init_gripper()
        except Exception as e:
            logger.error("[connect] Error connecting to robot %s: %s", robot_ip, e)
            raise

    def disconnect(self):
        """Close RTDE connections."""
        if self.dry_run:
            logger.info("[disconnect] dry-run: nothing to disconnect.")
            return
        if self.rtde_c is not None:
            self.rtde_c.stopScript()
            self.rtde_c.disconnect()
            logger.info("[disconnect] RTDEControlInterface disconnected.")
        if self.rtde_r is not None:
            self.rtde_r.disconnect()
            logger.info("[disconnect] RTDEReceiveInterface disconnected.")

    def _read_pose_from_json(self, file_path, pose_key):
        """Read the pose associated with ``pose_key`` from a JSON file."""
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            pose = data.get(str(pose_key), None)
            if pose is None:
                logger.error("The pose with key '%s' is not defined in %s.", pose_key, file_path)
            return pose
        except Exception as e:
            logger.error("Error reading JSON file %s: %s", file_path, e)
            return None

    def move_to_pose(self, pose, speed=None, acceleration=None):
        """
        Move the robot to ``pose`` ([x, y, z, rx, ry, rz]) after validating it
        against the workspace envelope. Speed/acceleration are clamped.
        """
        actual_speed = min(speed if speed is not None else self.speed, MAX_SPEED)
        actual_acceleration = min(
            acceleration if acceleration is not None else self.acceleration, MAX_ACCELERATION
        )
        if not self._validate_pose(pose):
            return
        logger.info("[move_to_pose] Move to %s (speed=%s, accel=%s)",
                    pose, actual_speed, actual_acceleration)
        if self.dry_run:
            logger.info("[move_to_pose] dry-run: motion not sent to robot.")
            return
        if self.rtde_c is None:
            logger.error("[move_to_pose] Invalid RTDE connection!")
            return
        try:
            self.rtde_c.moveL(pose, actual_speed, actual_acceleration)
        except Exception as e:
            logger.error("[move_to_pose] moveL failed: %s", e)
            try:
                self.rtde_c.stopL(2.0)
            except Exception:
                pass
            return
        time.sleep(self.move_delay)

    def move_to_grasping(self):
        """Read the grasping pose from the scene JSON and move towards it."""
        pose_file = str(SCENE_DIR / "sorted_tcp_poses.json")
        grasp_pose = self._read_pose_from_json(pose_file, self.default_grasp_pose_key)
        if grasp_pose is None:
            logger.error("Unable to perform move_to_grasping: pose not available.")
            return
        self.move_to_pose(grasp_pose)

    def move_to_target(self, _ignored_color=None):
        """
        Move to the target defined by the scene's target_pose.json (x, y only),
        keeping the current z/rx/ry/rz, then lower z by 1.5 cm.
        """
        target_json_path = str(SCENE_DIR / "target_pose.json")
        try:
            with open(target_json_path, "r") as f:
                data = json.load(f)
            # take the first detection and only x,y
            target_xy = data[0][:2]
        except Exception as e:
            logger.error("[move_to_target] Error reading target JSON file %s: %s",
                         target_json_path, e)
            return

        current_pose = self.get_current_tcp_pose()
        if current_pose is None:
            logger.error("[move_to_target] Unable to get current TCP pose.")
            return

        # 1) build and execute the usual move (swap in X,Y)
        new_pose = current_pose.copy()
        new_pose[0] = target_xy[0]
        new_pose[1] = target_xy[1]
        logger.info("[move_to_target] Move to the target (x,y): %s", new_pose)
        self.move_to_pose(new_pose)

        # 2) now lower Z by 1.5 cm and execute that move
        lowered_pose = new_pose.copy()
        lowered_pose[2] -= 0.015
        logger.info("[move_to_target] Lowering by 1.5 cm along Z: %s", lowered_pose)
        self.move_to_pose(lowered_pose)

    def move_to_home(self):
        """Move the robot to the home position."""
        logger.info("[move_to_home] Moving to home configuration.")
        self.move_to_pose(self._home_pose)

    def gripper_open(self):
        """Open the gripper."""
        if self.dry_run:
            logger.info("[gripper_open] dry-run: opening gripper (not sent).")
            return
        if self.gripper is not None:
            logger.info("[gripper_open] Opening the gripper.")
            self.gripper.open()
        else:
            logger.error("[gripper_open] gripper not initialized.")
        time.sleep(self.gripper_delay)

    def gripper_close(self):
        """Close the gripper."""
        if self.dry_run:
            logger.info("[gripper_close] dry-run: closing gripper (not sent).")
            return
        if self.gripper is not None:
            logger.info("[gripper_close] Closing the gripper.")
            self.gripper.close()
        else:
            logger.error("[gripper_close] gripper not initialized.")
        time.sleep(self.gripper_delay)

    def approach(self):
        """De-approach move: grasping pose from the scene JSON with z + 0.05."""
        pose_file = str(SCENE_DIR / "sorted_tcp_poses.json")
        grasp_pose = self._read_pose_from_json(pose_file, self.default_grasp_pose_key)
        if grasp_pose is None:
            logger.error("Unable to perform approach: grasping pose not available.")
            return
        deapproach_pose = grasp_pose.copy()
        deapproach_pose[2] += 0.05
        logger.info("[approach] De-approach movement towards the pose %s", deapproach_pose)
        self.move_to_pose(deapproach_pose)

    def get_current_tcp_pose(self):
        """Return the current TCP pose of the robot."""
        if self.dry_run:
            logger.info("[get_current_tcp_pose] dry-run: returning home pose as current.")
            return list(self._home_pose)
        if self.rtde_r is not None:
            current_pose = self.rtde_r.getActualTCPPose()
            logger.info("[get_current_tcp_pose] The current pose is: %s", current_pose)
            return current_pose
        logger.error("[get_current_tcp_pose] Invalid RTDE connection!")
        return None

    def positive_shift(self, axis, offset):
        """Positive shift of the current pose along ``axis`` by ``offset`` (m)."""
        current_pose = self.get_current_tcp_pose()
        if current_pose is None:
            logger.error("Unable to get current pose for positive_shift.")
            return
        new_pose = current_pose.copy()
        if axis.lower() == "x":
            new_pose[0] += offset
        elif axis.lower() == "y":
            new_pose[1] += offset
        elif axis.lower() == "z":
            new_pose[2] += offset
        else:
            logger.error("Axis '%s' not recognized for positive_shift.", axis)
            return
        logger.info("[positive_shift] New pose after positive shift on %s: %s", axis, new_pose)
        self.move_to_pose(new_pose)

    def negative_shift(self, axis, offset):
        """Negative shift of the current pose along ``axis`` by ``offset`` (m)."""
        current_pose = self.get_current_tcp_pose()
        if current_pose is None:
            logger.error("Unable to get current pose for negative_shift.")
            return
        new_pose = current_pose.copy()
        if axis.lower() == "x":
            new_pose[0] -= offset
        elif axis.lower() == "y":
            new_pose[1] -= offset
        elif axis.lower() == "z":
            new_pose[2] -= offset
        else:
            logger.error("Axis '%s' not recognized for negative_shift.", axis)
            return
        logger.info("[negative_shift] New pose after negative shift on %s: %s", axis, new_pose)
        self.move_to_pose(new_pose)
