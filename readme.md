# VL-GRiP3: A Hierarchical Pipeline Leveraging Vision-Language Models for Autonomous Robotic 3D Grasping

This repository represents the official implementation of the paper:
Paper: [Title of my paper](https://YOUR-LINK-HERE)


VL_GRiP3 is an end-to-end pipeline for **vision–language-driven grasping** on a UR5e robot with a Robotiq Hand-E gripper.

![VL-GRiP3](assets/banner.png)

This framework provides a transparent, modular pipeline that decomposes language understanding, perception, and action planning for robotic manipulation. A single vision–language model (VLM) backbone interprets natural-language commands, localizes the target, and produces high-level action intent. To handle occlusions from a single RGB-D view, CAD-augmented point cloud registration reconstructs a more complete 3D representation at low hardware cost. An M2T2-based grasp planner then predicts geometry-aware 3D grasp poses from the augmented point cloud, enabling reliable manipulation of irregular industrial parts in SME manufacturing settings.

High-level flow:

1. Capture RGB-D with the **Orbbec Gemini 2L**
2. Segment the commanded object + detect the target area with **PaliGemma**
3. Register the CAD model to the scene with **OverlapPredator**
4. Generate grasp candidates with **M2T2**
5. Decode a high-level action script from language + image (PaliGemma action head)
6. Execute the sequence on the UR5e via **RTDE**

> **Note (grasping, work in progress).** `GRiP3_Pipeline/utils/grasp_feasibility.py` is a
> standalone module that rejects physically infeasible M2T2 grasp candidates (gripper opening,
> approach cone, gripper–scene collision, UR5e reachability via `ur_ikfast`) before ranking.
> It is unit-testable on its own (`python GRiP3_Pipeline/utils/grasp_feasibility.py`) but is
> **not yet wired into** `class_m2t2`, which still selects the top grasp by raw confidence.

## Citation
If you find this code useful for your work or use it in your project, please consider citing:

```bash
@article{polonara2026vl,
  title={VL-GRiP3: A hierarchical pipeline leveraging vision-language models for autonomous robotic 3D grasping},
  author={Polonara, Mirco and Yang, Xingyu and Carbonari, Luca and Zhang, Xuping},
  journal={Robotics and Computer-Integrated Manufacturing},
  volume={100},
  pages={103244},
  year={2026},
  publisher={Elsevier}
}
```

## Installation
This code has been tested on

- Python 3.10.19, PyTorch 2.3.1+cu121, CUDA 12.1, NVIDIA RTX 4090 (24 GB VRAM)

Target robot platform:

- **UR5e** arm controlled over **RTDE**
- **Robotiq Hand-E** parallel-jaw gripper (~50 mm stroke)
- **Orbbec Gemini 2L** RGB-D camera (via the `pyorbbecsdk` Python bindings)

All paths are resolved relative to the repository through `GRiP3_Pipeline/utils/paths.py`,
so the pipeline runs from any working directory or machine without editing the code. You can
optionally override the project root or the active scene directory with the `VLGRIP3_ROOT` and
`VLGRIP3_SAMPLE_DIR` environment variables.


## Requirements
Clone and follow the official setup instructions for:

- [OverlapPredator](https://github.com/prs-eth/OverlapPredator)
- [M2T2](https://github.com/NVlabs/M2T2)

Clone our repository and install the dependencies:

```bash
git clone https://github.com/AU-DK-Robotics/VL-GRiP3.git
cd VL-GRiP3
pip install -r requirements.txt
```

`requirements.txt` installs the core stack (PyTorch + CUDA 12.1 wheels) and the bundled
`M2T2/pointnet2_ops`. A few hardware/feature-specific dependencies are intentionally left
commented out at the end of the file — install them only if you need the corresponding feature:

- `pyorbbecsdk` — Orbbec Gemini 2L camera capture (`OrbbecCapture`)
- `openai-whisper` and `sounddevice` — spoken-command input (`main_whisper.py`)

## Checkpoints
Create your dataset and train Paligemma, then place your trained weights in:
```bash
cd GRiP3_Pipeline/checkpoints

```

Create your dataset and train OverlapPredator, then place your trained weights in:

```bash
cd OverlapPredator/weights
```


## Run
After creating the virtual environment, preparing the dataset, and training PaliGemma and Predator, VL-GRiP3 can be run with::

```bash

python main.py
```
If you want to use OpenAI Whisper module, please run:
```bash
python main_whisper.py
```

Example command and repository directory tree:

```text
move tac1 in red target
move tac2 in green target
move tac3 in yellow target


VL-GRiP3/
├─ main.py                            # End-to-end pipeline (typed prompt)
├─ main_whisper.py                    # End-to-end pipeline (spoken prompt via Whisper)
├─ requirements.txt
├─ M2T2/                              # Vendored M2T2 (includes pointnet2_ops, installed locally)
├─ OverlapPredator/                   # Vendored OverlapPredator (CAD-to-scene registration)
│  ├─ configs/
│  │  ├─ test/vl_grip3_tac{1,2,3}.yaml  # Per-object demo configs (repo-relative paths)
│  │  └─ train/vlam.yaml                # Training config
│  ├─ scripts/demo_save.py              # Registration entry point (run as a subprocess)
│  ├─ build_training_file.py           # Builds train/val pairs from CAD + scans
│  ├─ weights/                          # (user-created) trained Predator weights
│  └─ training/                         # CAD models + per-object registration pairs
│
└─ GRiP3_Pipeline/
   ├─ ft_action_module.py             # Fine-tune the action (command) PaliGemma head
   ├─ ft_segm_module.py               # Fine-tune the segmentation PaliGemma head
   ├─ checkpoints/                    # (user-created) trained PaliGemma adapters (PEFT)
   │  ├─ paligemma-segm-module/       # Segmentation head weights
   │  └─ paligemma-action-module/     # Action-command head weights
   │
   ├─ sample_data/
   │  └─ real_world/
   │     └─ MX/                       # Active scene dir (capture inputs + module outputs)
   │        ├─ rgb.png                # Captured RGB (Orbbec Gemini 2L)
   │        ├─ depth.npy              # Captured depth in metres (Orbbec Gemini 2L)
   │        ├─ meta_data.pkl          # Camera intrinsics/extrinsics for M2T2
   │        ├─ seg.png                # Segmentation mask (PaliGemma)
   │        ├─ segmentation_overlay.png
   │        ├─ detection_target_overlay.png
   │        ├─ target_pose.json       # Target location in the robot frame
   │        ├─ scene.pth              # Scene point cloud for OverlapPredator
   │        ├─ full_object_pointcloud.npz  # Registered object+scene cloud for M2T2
   │        ├─ best_poses.json        # Best grasps/placements (M2T2)
   │        └─ sorted_tcp_poses.json  # Ranked TCP poses for UR5eCommands
   │
   └─ utils/
      ├─ paths.py                     # Central, location-independent path resolution
      ├─ class_get_snap_mm.py         # OrbbecCapture (Gemini 2L RGB-D grabber)
      ├─ class_paligemma.py           # PaliGemmaInference (segment / detect / command)
      ├─ class_predator.py            # PredatorPipeline (scene.pth + OverlapPredator + NPZ)
      ├─ class_m2t2.py                # M2T2Inference (grasp / placement planning)
      ├─ grasp_feasibility.py         # Standalone physical-feasibility filter for grasps (UR5e + Hand-E)
      ├─ class_robot_library.py       # UR5eCommands + RobotiqGripper (Hand-E, RTDE control)
      ├─ class_whisper.py             # WhisperTranscriber (speech-to-text)
      ├─ robotiq_gripper_control.py   # Robotiq gripper low-level control
      ├─ robotiq_preamble.py          # Robotiq URScript preamble
      ├─ config.yaml                  # M2T2 inference config
      ├─ m2t2/                        # Vendored M2T2 inference code
      ├─ big_vision/                  # PaliGemma helper code
      ├─ dataset/                     # PaliGemma fine-tuning data (segmentation + commands)
      └─ whisper/                     # Whisper scratch dir (recorded prompt.wav)
```

## Sample Data

For an overview of the required inputs and the outputs of each module, see:

```bash
cd GRiP3_Pipeline/sample_data/real_world/MX
```

This folder contains example M2T2 inputs and module outputs, including object/target identification and grasp pose estimation.



## Acknowledgments

In this project we use (parts of) the official implementations of the following works:

- [OverlapPredator](https://github.com/prs-eth/OverlapPredator)
- [M2T2](https://github.com/NVlabs/M2T2)

We thank the respective authors for open-sourcing their methods. We would also like to thank reviewers for their valuable inputs.



