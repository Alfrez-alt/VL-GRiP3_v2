"""VL-GRiP3 pipeline entry point.

Captures an RGB-D scene, grounds the commanded object with PaliGemma, registers
the CAD model with OverlapPredator, plans grasps with M2T2, decodes a high-level
action script and (optionally) executes it on the UR robot.

Examples:
    python main.py                         # full pipeline, keyboard prompt
    python main.py --voice                 # use Whisper speech-to-text
    python main.py --no-capture            # reuse existing scene files
    python main.py --no-robot              # dry-run: never command the robot
"""
import argparse
import logging
import os
import re
import sys
import time

logger = logging.getLogger("vl_grip3.main")


def process_user_prompt(user_prompt: str):
    # 1) Extract the command verb and object for segmentation.
    object_match = re.search(
        r"(?P<verb>move|place|put|position|shift|locate)\s+(?:the\s+)?(?P<object>.+?)(?=\s+(?:to|in|on|into|at|positive|negative)\b|$)",
        user_prompt,
        re.IGNORECASE
    )
    if object_match:
        verb = object_match.group("verb").lower()
        object_phrase = object_match.group("object").strip()
    else:
        raise ValueError("Could not extract a valid object for segmentation.")

    object_seg_prompt = "<image> segment " + object_phrase.lower()

    # 2) Check for shift commands.
    direction_match = re.search(
        r"\b(positive|negative)\s+([xyz])(?:\s+by\s+([-+]?[0-9]*\.?[0-9]+))?\b",
        user_prompt,
        re.IGNORECASE
    )
    if direction_match:
        label = direction_match.group(1).lower()
        direction = direction_match.group(2).lower()
        offset = direction_match.group(3)
        if offset:
            command_prompt = f"<image> shift {label} {direction} by {offset}"
        else:
            command_prompt = f"<image> shift {label} {direction}"
        return object_seg_prompt, None, command_prompt

    # 3) Look for a target location pattern.
    target_match = re.search(
        r"\b(?:to|in|on|into|at|inside)\s+(?:the\s+)?(?P<color>\w+)\s+(?P<target>\w+)",
        user_prompt,
        re.IGNORECASE
    )
    if target_match:
        color = target_match.group("color").lower()
        target_type = target_match.group("target").lower()
        target_seg_prompt = f"<image> detect {color} {target_type}"
        command_prompt = f"<image> move in {color} target"
        return object_seg_prompt, target_seg_prompt, command_prompt

    # 4) Fallback.
    command_remainder = user_prompt[object_match.end():].strip()
    if not command_remainder:
        raise ValueError("The command part after the object could not be extracted.")
    if not command_remainder.lower().startswith(verb):
        command_prompt = f"<image> {verb} " + command_remainder
    else:
        command_prompt = "<image> " + command_remainder

    return object_seg_prompt, None, command_prompt


def get_user_prompt(use_voice: bool) -> str:
    """Acquire the natural-language command from keyboard or Whisper."""
    if use_voice:
        from GRiP3_Pipeline.utils.class_whisper import WhisperTranscriber
        transcriber = WhisperTranscriber()
        while True:
            user_prompt = transcriber.record_and_transcribe()
            logger.info("Transcribed prompt: %s", user_prompt)
            if input("If this transcription is OK, press Enter; otherwise type 'R' to re-record: ").strip().lower() != 'r':
                return user_prompt
    return input("Enter your prompt (e.g. 'move tac2 in red target'): ")


def execute_command_sequence(ur3, command_line: str):
    """Map the decoded high-level command string onto UR3Commands calls."""
    command_mapping = {
        "gripper open": ur3.gripper_open,
        "gripper close": ur3.gripper_close,
        "grasp": ur3.move_to_grasping,
        "approach": ur3.approach,
        "home pose": ur3.move_to_home,
    }
    commands = [cmd.strip() for cmd in command_line.split(';') if cmd.strip()]

    for cmd in commands:
        normalized_cmd = cmd.lower().strip()

        if normalized_cmd == "connect":
            logger.info("Running the command: connect")
            ur3.connect()
        elif normalized_cmd == "disconnect":
            logger.info("Executing the command: disconnect")
            ur3.disconnect()
        elif normalized_cmd.startswith("positive(") or normalized_cmd.startswith("negative("):
            pattern_shift = r"^(positive|negative)\(\s*([xyz])\s*,\s*([\d\.]+)\s*\)$"
            match_shift = re.match(pattern_shift, normalized_cmd, re.IGNORECASE)
            if match_shift:
                shift_type, axis, offset_str = match_shift.groups()
                try:
                    offset = float(offset_str)
                except ValueError:
                    logger.warning("Invalid offset value in: %s", cmd)
                    continue
                logger.info("Executing the command: %s", cmd)
                if shift_type.lower() == "positive":
                    ur3.positive_shift(axis, offset)
                else:
                    ur3.negative_shift(axis, offset)
            else:
                logger.warning("Invalid shift command format: %s", cmd)
        elif normalized_cmd.startswith("target("):
            logger.info("Executing the target command (ignoring parameter)")
            ur3.move_to_target()
        elif normalized_cmd == "target":
            logger.info("Executing the target command")
            ur3.move_to_target()
        elif normalized_cmd in command_mapping:
            logger.info("Executing the command: %s", cmd)
            command_mapping[normalized_cmd]()
        else:
            logger.warning("Command '%s' not recognized.", cmd)

        time.sleep(0.1)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="VL-GRiP3 vision-language grasping pipeline")
    p.add_argument("--voice", action="store_true",
                   help="use Whisper speech-to-text for the prompt")
    p.add_argument("--scene-dir", default=None,
                   help="scene/working directory (overrides VLGRIP3_SCENE_DIR)")
    p.add_argument("--no-capture", action="store_true",
                   help="skip RealSense capture and reuse existing scene files")
    p.add_argument("--no-robot", action="store_true",
                   help="dry-run: run the full pipeline but never command the robot")
    p.add_argument("--log-level", default=None,
                   help="logging level (DEBUG/INFO/WARNING/...)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # Propagate the scene-dir override BEFORE importing modules that resolve it.
    if args.scene_dir:
        os.environ["VLGRIP3_SCENE_DIR"] = args.scene_dir

    from GRiP3_Pipeline.utils.logging_config import setup_logging
    setup_logging(args.log_level)

    from GRiP3_Pipeline.utils import paths
    from GRiP3_Pipeline.utils.class_m2t2 import M2T2Inference
    from GRiP3_Pipeline.utils.class_robot_library import UR3Commands
    from GRiP3_Pipeline.utils.class_predator import PredatorPipeline
    from GRiP3_Pipeline.utils.class_paligemma import PaliGemmaInference
    import torch

    scene_dir = str(paths.SCENE_DIR)
    base_model_id = "google/paligemma-3b-mix-448"
    classes = ["cylinder", "cube", "prism", "triangle"]

    # 1. Capture the scene using RealSense (unless reusing existing files).
    if args.no_capture:
        logger.info("Skipping capture; reusing scene files in %s", scene_dir)
    else:
        from GRiP3_Pipeline.utils.class_get_snap_mm import RealSenseCapture
        capture = RealSenseCapture(
            save_directory=scene_dir,
            rgb_filename="rgb.png",
            depth_filename="depth.npy",
            resolution_width=640,
            resolution_height=480,
            fps=30,
        )
        logger.info("Starting scene capture...")
        capture.run_capture()

    # 2. Acquire and parse the user prompt.
    user_prompt = get_user_prompt(args.voice)
    logger.info("Prompt received: %s", user_prompt)
    seg_prompt, seg_target_prompt, cmd_prompt = process_user_prompt(user_prompt)
    logger.info("Segmentation prompt: %s", seg_prompt)
    logger.info("Segmentation target prompt: %s", seg_target_prompt)
    logger.info("Command prompt: %s", cmd_prompt)

    # Decide which TAC object to use for registration (tac1/tac2/tac3).
    m = re.search(r"\b(tac1|tac2|tac3)\b", user_prompt, re.IGNORECASE)
    if not m:
        logger.error("No 'tac1', 'tac2' or 'tac3' found in the prompt. "
                     "Specify which tac to move, e.g.: 'move tac2 in red target'.")
        sys.exit(1)
    cad_name = m.group(1).lower()
    logger.info("Detected object for registration: %s", cad_name)

    image_path = os.path.join(scene_dir, "rgb.png")
    image_path_action_inf = str(paths.ACTION_INFER_IMAGE)
    seg_output_path = os.path.join(scene_dir, "segmentation_overlay.png")
    seg_output_path_target = os.path.join(scene_dir, "detection_target_overlay.png")

    # 3. Segmentation inference.
    inferencer = PaliGemmaInference(
        peft_model_path=str(paths.SEGM_CHECKPOINT),
        base_model_id=base_model_id,
    )
    logger.info("Running inference for segmentation...")
    inferencer.infer(prompt=seg_prompt, image_path=image_path,
                     output_path=seg_output_path, classes=classes, mode="segmentation")
    if seg_target_prompt:
        inferencer.infer(prompt=seg_target_prompt, image_path=image_path,
                         output_path=seg_output_path_target, classes=classes, mode="detect")
    del inferencer
    torch.cuda.empty_cache()
    logger.info("Segmentation model unloaded. GPU cache cleared.")

    # 4. Registration via PredatorPipeline.
    input("Press Enter to proceed with registration...")
    predator_cfg_path = str(paths.PREDATOR_CFG_DIR / f"vl_grip3_{cad_name}.yaml")
    logger.info("Using Predator config: %s", predator_cfg_path)
    while True:
        pipeline = PredatorPipeline(
            sample_dir=scene_dir,
            predator_script=str(paths.PREDATOR_SCRIPT),
            predator_cfg=predator_cfg_path,
            label=101,
        )
        pipeline.run()
        logger.info("Registration completed. NPZ available in: %s", pipeline.npz_path)
        if input("Type 'R' to repeat or press Enter to continue: ").strip().lower() != 'r':
            break

    # 5. Grasping (M2T2).
    input("Press Enter to proceed with M2T2 inference:")
    while True:
        M2T2Inference().run()
        if input("If the visualization is satisfactory, press Enter; otherwise type 'R' to repeat: ").strip().lower() != 'r':
            break

    # 6. Command (action) inference.
    inferencer_cmd = PaliGemmaInference(
        peft_model_path=str(paths.ACTION_CHECKPOINT),
        base_model_id=base_model_id,
    )
    logger.info("Running inference for command...")
    cmd_detections = inferencer_cmd.infer(
        prompt=cmd_prompt, image_path=image_path_action_inf, classes=classes, mode="command"
    )

    input("Press Enter to proceed with robot execution...")

    # 7. Execute the command on the robot (dry-run when --no-robot).
    ur3 = UR3Commands(dry_run=args.no_robot)
    command_line = cmd_detections.split('\n')[-1].strip()
    logger.info("Sequence of commands to execute: %s", command_line)
    try:
        execute_command_sequence(ur3, command_line)
    except Exception as e:
        logger.error("Robot execution aborted: %s", e)
        try:
            ur3.disconnect()
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
