<p align="center">
    <img src="./docs/gigpo/logo-SERL.png" alt="SERL logo" width="50%">
</p>

# SERL

## What and When to Distill: Selective Hindsight Distillation for Multi-Turn Agents

SERL is the official codebase for **Selective Environment-Reweighted Learning**, a framework for using hindsight feedback in long-horizon LLM agent reinforcement learning.

Paper: [SERL4nips.pdf](./SERL4nips.pdf)  
Status: submitted to NeurIPS 2026

Long-horizon agent training has a credit-assignment problem: a sparse episode reward tells us whether the whole rollout succeeds, but not which action in a multi-turn trajectory caused success or failure. Agent environments already return useful feedback after each action, such as an error message, a changed page, a new observation, or a successful reference trajectory. SERL studies how to use these signals without turning privileged hindsight into unstable imitation.

The central principle is:

> The task reward determines the update direction; hindsight feedback only adjusts where and how strongly that update is applied.

SERL implements this by using an environment-conditioned teacher to reweight the GRPO objective on executable action tokens. The teacher sees training-only hindsight, the student does not. The resulting teacher-student likelihood gap is clipped, sign-aware, decayed during training, and masked to action spans so that feedback improves credit assignment without controlling the full policy update.

## Highlights

- Systematic study of feedback source and feedback placement for multi-turn LLM agents.
- SERL objective built on GRPO with bounded, action-token-level hindsight reweighting.
- Step-level and anchor-level feedback placement.
- Support for immediate feedback, next observation, future trajectory, successful trajectory combinations, and LLM-judged trajectory feedback.
- Reproduction scripts for ALFWorld and WebShop with Qwen2.5-7B-Instruct.

## Repository Layout

```text
recipe/serl/                         SERL training recipe, config, and launch scripts
recipe/serl/run_alfworld.sh          ALFWorld reproduction script
recipe/serl/run_webshop.sh           WebShop reproduction script
agent_system/environments/           Multi-turn agent environment wrappers
judge_utils/                         Utilities for LLM-judged feedback
examples/data_preprocess/prepare.py  Text-mode parquet preparation
SERL4nips.pdf                        NeurIPS 2026 submission draft
```

## Main Results

The paper trains Qwen2.5-7B-Instruct on ALFWorld and WebShop. All RL methods use group size 8 and the same main training budget.

| Method | ALFWorld Avg. Success | WebShop Score | WebShop Success |
| --- | ---: | ---: | ---: |
| GRPO | 75.3 | 73.1 | 64.1 |
| GIGPO | 83.9 | 83.5 | 75.8 |
| HGPO | 85.8 | 88.4 | 77.8 |
| RLSD | 82.9 | 83.6 | 75.8 |
| **SERL** | **90.0** | **89.5** | **80.1** |

Relative to GRPO, SERL improves ALFWorld average success by 14.7 points and WebShop success by 16.0 points. Relative to the strongest pure RL baseline, HGPO, SERL improves ALFWorld by 4.2 points and WebShop success by 2.3 points.

## Method Overview

SERL separates two questions:

1. **What should the teacher see?**  
   The teacher can condition on immediate feedback, next observations, future trajectories, successful trajectories, current trajectories, or combinations of these signals.

2. **When should that feedback affect learning?**  
   Feedback can be applied at every transition (`step-level`) or concentrated around semantically meaningful state groups (`anchor-level`).

For an action token, SERL compares the student probability with a stop-gradient teacher probability conditioned on placed hindsight. The teacher-student log-probability gap becomes a bounded coefficient on the GRPO advantage. Action tokens receive the reweighted advantage; reasoning and formatting tokens keep the original reward-driven GRPO update.

The teacher weight decays during training. Early training uses hindsight to improve credit assignment when exploration is weak; later training returns control to the task reward to reduce privileged-information bias.

## Feedback Modes

Set the feedback source with `SAMPLING_MODE=<mode>`. The scripts default to `immediate_feedback`.

Implementation names use `successful_sample` for the paper's successful trajectory.

| `SAMPLING_MODE` | Paper feedback source | ALFWorld Avg. | WebShop Score | WebShop Success |
| --- | --- | ---: | ---: | ---: |
| `immediate_feedback` | immediate feedback | 90.0 | 89.5 | 80.1 |
| `next_observation` | next observation | 76.6 | 90.5 | 77.7 |
| `future_trajectory` | future trajectory | 83.1 | 85.9 | 76.6 |
| `successful_sample_or_immediate_feedback` | successful trajectory or immediate feedback | 81.9 | 84.1 | 72.7 |
| `successful_sample_immediate_feedback` | successful trajectory and immediate feedback | **90.4** | 87.7 | **81.3** |
| `successful_sample_next_observation` | successful trajectory and next observation | 85.6 | 86.9 | 77.7 |
| `successful_sample_future_trajectory` | successful trajectory and future trajectory | 83.3 | 88.1 | 76.6 |
| `successful_sample_future_trajectory_immediate_feedback` | successful trajectory, future trajectory, and immediate feedback | 76.1 | 87.1 | 77.0 |
| `successful_sample_future_trajectory_next_observation` | successful trajectory, future trajectory, and next observation | 76.6 | 84.1 | 76.2 |

Examples:

```bash
SAMPLING_MODE=immediate_feedback bash recipe/serl/run_alfworld.sh
SAMPLING_MODE=successful_sample_immediate_feedback bash recipe/serl/run_webshop.sh
SAMPLING_MODE=successful_sample_future_trajectory_next_observation bash recipe/serl/run_webshop.sh
```

## Anchor Modes

Anchor-level placement groups semantically similar states and applies hindsight around meaningful state changes. In the code, anchor is enabled by using the `anchor_` prefix. To disable anchor, use the corresponding non-anchor mode.

Supported anchor modes:

```text
anchor_immediate_feedback
anchor_next_observation
anchor_future_trajectory
anchor_successful_sample_or_immediate_feedback
anchor_successful_sample_immediate_feedback
anchor_successful_sample_next_observation
anchor_successful_sample_future_trajectory
anchor_successful_sample_future_trajectory_immediate_feedback
anchor_successful_sample_future_trajectory_next_observation
```

Examples:

```bash
SAMPLING_MODE=anchor_immediate_feedback bash recipe/serl/run_alfworld.sh
SAMPLING_MODE=anchor_successful_sample_immediate_feedback bash recipe/serl/run_webshop.sh
```

Optional similarity filtering can be enabled with Hydra overrides:

```bash
SAMPLING_MODE=anchor_immediate_feedback \
bash recipe/serl/run_webshop.sh \
  actor_rollout_ref.actor.serl.anchor_enable_similarity=True \
  actor_rollout_ref.actor.serl.anchor_similarity_thresh=0.95
```

In the WebShop ablation, anchor placement is most useful when it suppresses redundant or weakly causal dense feedback:

| Feedback source | Step Success | Anchor Success |
| --- | ---: | ---: |
| immediate feedback | 80.1 | 79.3 |
| next observation | 77.7 | 76.2 |
| future trajectory | 76.6 | 75.4 |
| successful trajectory or immediate feedback | 72.7 | 74.8 |
| successful trajectory and immediate feedback | 81.3 | **81.9** |
| successful trajectory and next observation | 77.7 | 79.7 |
| successful trajectory and future trajectory | 76.6 | 77.7 |
| successful trajectory, future trajectory, and immediate feedback | 77.0 | 79.3 |
| successful trajectory, future trajectory, and next observation | 72.9 | 76.3 |

## LLM-Judged Feedback

SERL also supports judged feedback, where an external OpenAI-compatible judge summarizes a trajectory into concise privileged guidance before teacher scoring.

Supported modes:

| `SAMPLING_MODE` | Meaning |
| --- | --- |
| `judge_current_traj` | Judge the current trajectory. |
| `judge_current_traj_on_successful_sample` | Judge the current trajectory with a successful trajectory as reference. |

Example:

```bash
JUDGE_API_URL=http://localhost:8000/v1 \
JUDGE_MODEL=your-judge-model \
JUDGE_API_KEY=your-api-key \
SAMPLING_MODE=judge_current_traj \
bash recipe/serl/run_alfworld.sh
```

With Kimi-K2.6 as judge, the paper reports:

| Judge feedback | ALFWorld Avg. | WebShop Score | WebShop Success |
| --- | ---: | ---: | ---: |
| Current Trajectory | 86.7 | 88.2 | 78.1 |
| Current Trajectory + Successful Trajectory | 89.4 | 89.0 | 81.8 |

The paper also reports that judge capacity matters: Qwen2.5-7B-Instruct gives useful signal on ALFWorld but is weaker on WebShop, likely because WebShop trajectories contain long HTML-like observations and require a longer context window.

## Trajectory Format

SERL supports two trajectory organization formats:

| Format | Description |
| --- | --- |
| `response` | Response-oriented trajectory rendering. This is the default. |
| `observation_action` | Observation-action turn rendering. |

Choose the format with `TRAJECTORY_FORMAT=<format>`:

```bash
TRAJECTORY_FORMAT=response bash recipe/serl/run_alfworld.sh
TRAJECTORY_FORMAT=observation_action bash recipe/serl/run_webshop.sh
```

## Installation

### Base Runtime

Create the base SERL environment from the repository root:

```bash
conda create -n serl python==3.12 -y
conda activate serl

pip3 install vllm==0.11.0
pip3 install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
pip install -e .
```

Environment packages may have conflicting Python and dependency requirements. Use a separate conda environment for each environment backend when needed.

### ALFWorld

Install ALFWorld:

```bash
pip3 install gymnasium==0.29.1
pip3 install stable-baselines3==2.6.0
pip install alfworld
```

Download PDDL files, game files, and the pretrained MaskRCNN detector:

```bash
alfworld-download -f
```

Use `--extra` if you also want pretrained checkpoints and seq2seq data:

```bash
alfworld-download -f --extra
```

Verify the text game installation:

```bash
alfworld-play-tw
```

### WebShop

WebShop requires Python `<=3.10`, so create a dedicated environment:

```bash
conda create -n serl-webshop python==3.10 -y
conda activate serl-webshop
```

Install WebShop data and dependencies:

```bash
cd ./agent_system/environments/env_package/webshop/webshop
./setup.sh -d all
```

If `gdown` fails, visit `https://drive.google.com/`, get your Google Drive cookie, and paste it into `.cache/gdown/cookies.txt`. Manual download of the required files is also acceptable.

After WebShop is installed, return to the SERL repository root and install the training dependencies in the same `serl-webshop` environment:

```bash
cd /path/to/SERL
pip3 install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn==2.7.4.post1 --no-build-isolation
pip3 install -e .
pip3 install vllm==0.8.2
```

Warnings about `spacy` or `weasel` requiring an older `typer` can be ignored for the WebShop training scripts.

## Quickstart

### Prepare Text Data

The parquet files provide the text modality marker and dataset size. The actual task, observation, admissible actions, reward, and feedback are produced by the environment during rollout.

```bash
mkdir -p ~/data/serl/text
python3 examples/data_preprocess/prepare.py \
  --mode text \
  --local_dir ~/data/serl \
  --train_data_size 256 \
  --val_data_size 256
```

This creates:

```text
~/data/serl/text/train.parquet
~/data/serl/text/test.parquet
```

### Reproduce ALFWorld

```bash
conda activate serl
bash recipe/serl/run_alfworld.sh
```

Common overrides:

```bash
MODEL_PATH=Qwen/Qwen2.5-7B-Instruct \
TRAIN_FILE=~/data/serl/text/train.parquet \
VAL_FILE=~/data/serl/text/test.parquet \
OUTPUT_ROOT=./outputs/alfworld \
SAMPLING_MODE=immediate_feedback \
TRAJECTORY_FORMAT=response \
bash recipe/serl/run_alfworld.sh
```

### Reproduce WebShop

```bash
conda activate serl-webshop
bash recipe/serl/run_webshop.sh
```

Common overrides:

```bash
MODEL_PATH=Qwen/Qwen2.5-7B-Instruct \
TRAIN_FILE=~/data/serl/text/train.parquet \
VAL_FILE=~/data/serl/text/test.parquet \
OUTPUT_ROOT=./outputs/webshop \
SAMPLING_MODE=immediate_feedback \
TRAJECTORY_FORMAT=response \
bash recipe/serl/run_webshop.sh
```

The first positional argument can switch the rollout engine:

```bash
bash recipe/serl/run_alfworld.sh vllm
bash recipe/serl/run_webshop.sh vllm
```

Arbitrary Hydra overrides can be appended after the script:

```bash
SAMPLING_MODE=anchor_successful_sample_immediate_feedback \
bash recipe/serl/run_webshop.sh \
  trainer.total_epochs=150 \
  actor_rollout_ref.actor.optim.lr=1e-6
```

## Default Training Settings

The reproduction scripts follow the paper's main setup:

| Setting | ALFWorld | WebShop |
| --- | ---: | ---: |
| Base model | Qwen2.5-7B-Instruct | Qwen2.5-7B-Instruct |
| Rollout group size | 8 | 8 |
| Learning rate | `1e-6` | `1e-6` |
| Max environment steps | 50 | 15 |
| PPO mini-batch size | 256 | 64 |
| PPO micro-batch size per GPU | 32 | 8 |
| Initial distillation coefficient | 0.5 | 0.5 |
| Decay steps | 50 | 50 |
| Weight clip | 0.2 | 0.2 |
| Teacher sync interval | 10 | 10 |

The scripts expose common settings through environment variables:

| Variable | Default |
| --- | --- |
| `MODEL_PATH` | `Qwen/Qwen2.5-7B-Instruct` |
| `TRAIN_FILE` | `~/data/serl/text/train.parquet` |
| `VAL_FILE` | `~/data/serl/text/test.parquet` |
| `OUTPUT_ROOT` | `./outputs/<env>` |
| `SAMPLING_MODE` | `immediate_feedback` |
| `TRAJECTORY_FORMAT` | `response` |
| `N_GPUS_PER_NODE` | `8` |
| `TENSOR_MODEL_PARALLEL_SIZE` | `2` |
| `GROUP_SIZE` | `8` |

## Citation

The paper is currently an anonymous NeurIPS 2026 submission. Citation information will be updated after de-anonymization.

```bibtex
@misc{serl2026selective,
  title        = {What and When to Distill: Selective Hindsight Distillation for Multi-Turn Agents},
  year         = {2026},
  note         = {Submitted to NeurIPS 2026}
}
```

## Acknowledgement

SERL is implemented on top of [veRL](https://github.com/volcengine/verl). The environment integrations build on [ALFWorld](https://github.com/alfworld/alfworld) and [WebShop](https://github.com/princeton-nlp/WebShop). We thank the authors and contributors of these projects.
