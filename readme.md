# VL-GRiP3: A Hierarchical Pipeline Leveraging Vision-Language Models for Autonomous Robotic 3D Grasping

This repository represents the official implementation of the paper:
Paper: [Title of my paper](https://YOUR-LINK-HERE)


VL_GRiP3 is an end-to-end pipeline for **vision–language-driven grasping** on a UR3 robot with a Robotiq gripper.

![VL-GRiP3](assets/banner.png)

This framework provides a transparent, modular pipeline that decomposes language understanding, perception, and action planning for robotic manipulation. A single vision–language model (VLM) backbone interprets natural-language commands, localizes the target, and produces high-level action intent. To handle occlusions from a single RGB-D view, CAD-augmented point cloud registration reconstructs a more complete 3D representation at low hardware cost. An M2T2-based grasp planner then predicts geometry-aware 3D grasp poses from the augmented point cloud, enabling reliable manipulation of irregular industrial parts in SME manufacturing settings.

High-level flow:

1. Capture RGB-D with **RealSense**
2. Segment the commanded object + detect the target area with **PaliGemma**
3. Register the CAD model to the scene with **OverlapPredator**
4. Generate grasp candidates with **M2T2**
5. Decode a high-level action script from language + image (PaliGemma action head)
6. Execute the sequence on the UR3 via **RTDE**

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

- Python 3.10.19, PyTorch 2.3.1+cu121, CUDA 12.1, NVIDIA RTX 4090 (24GB VRAM)


## Requirements
Clone and follow the official setup instructions for:

- [OverlapPredator](https://github.com/prs-eth/OverlapPredator)
- [M2T2](https://github.com/NVlabs/M2T2)

Set up a **Python 3.10** environment (the stack is pinned around 3.10; on Ubuntu
24.04, 3.10 is available via the `deadsnakes` PPA):

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev

git clone <your-fork-url> VL-GRiP3_v2
cd VL-GRiP3_v2
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# 1) PyTorch matched to your CUDA toolkit (tested: CUDA 12.1)
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121

# 2) the rest of the dependencies
pip install -r requirements.txt

# 3) the pointnet2_ops CUDA extension (built from the vendored M2T2 copy)
pip install ./M2T2/pointnet2_ops
```

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
After creating the virtual environment, preparing the dataset, and training PaliGemma and Predator, VL-GRiP3 can be run with:

```bash
python main.py
```

Useful command-line flags:

```bash
python main.py --voice         # acquire the prompt via OpenAI Whisper (speech-to-text)
python main.py --no-capture    # skip RealSense capture and reuse existing scene files
python main.py --no-robot      # DRY-RUN: run the full pipeline but never command the robot
python main.py --log-level DEBUG
python main_whisper.py         # thin shim, equivalent to: python main.py --voice
```

### Configuration (no hardcoded paths)

Paths are resolved relative to the repository root, so the project runs from any
clone location. Override the defaults with environment variables when needed:

| Variable | Purpose | Default |
|---|---|---|
| `VLGRIP3_SCENE_DIR` | active scene / working directory | bundled `sample_data/real_world/MX` |
| `VLGRIP3_ROBOT_IP`  | UR controller IP | `192.168.1.254` |
| `VLGRIP3_CHECKPOINTS` | PaliGemma PEFT checkpoints dir | `GRiP3_Pipeline/checkpoints` |
| `VLGRIP3_PREDATOR` | OverlapPredator install dir | `OverlapPredator/` |
| `VLGRIP3_LOGLEVEL` | logging verbosity | `INFO` |

> **Robot safety / different arm:** the bundled target/home poses, workspace
> bounds and IP were calibrated for the original **UR3** cell. If you use a
> different arm (e.g. a **UR5e**), re-measure the poses and adjust
> `workspace_bounds` / `ROBOT_IP`. Motions are validated against a workspace
> envelope and speed/acceleration are clamped; use `--no-robot` to dry-run first.

Example command and repository directory tree:

```text
move tac1 in red target
move tac2 in green target
move tac3 in yellow target


GRiP3_Pipeline/
├─ checkpoints/
│  ├─ paligemma-segm-module/       # Segmentation head weights (PEFT)
│  └─ paligemma-action-module/     # Action command head weights (PEFT)
│  
│
├─ sample_data/
│  └─ real_world/
│     └─ MX/
│        ├─ rgb.png                # Captured RGB (RealSense)
│        ├─ depth.npy              # Captured depth (RealSense)
│        ├─ meta_data.pkl          # Camera intrinsics, extrinsics, etc. for M2T2
│        ├─ seg.png                # Segmentation mask (PaliGemma)
│        ├─ segmentation_overlay.png
│        ├─ detection_target_overlay.png
│        ├─ target_pose.json       # Target MX in robot frame
│        ├─ scene.pth              # Scene point cloud for Predator
│        ├─ full_object_pointcloud.npz
│        │                         # Registered object+scene cloud for M2T2
│        ├─ best_poses.json        # Best grasps/placements (M2T2)
│        └─ sorted_tcp_poses.json  # Sorted TCP poses for UR3Commands
│
└─ utils/
   ├─ class_get_snap_mm.py         # RealSenseCapture (RGB + depth grabber)
   ├─ class_paligemma.py           # PaliGemmaInference (seg/detect/command)
   ├─ class_predator.py            # PredatorPipeline (scene.pth + OverlapPredator + NPZ)
   ├─ class_m2t2.py                # M2T2Inference (loads config.yaml + runs grasping)
   ├─ class_robot_library.py       # UR3Commands + RobotiqGripper (RTDE control)
   ├─ class_whisper                # Whisper class
   ├─ ft_action_module.py          # Script to fine-tune the action (command) PaliGemma head
   └─ ft_segm_module.py            # Script to fine-tune the segmentation PaliGemma head
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



