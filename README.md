# Isaac SO-ARM101 RL for VLA Pen Pickup

## 1. Task Philosophy
The `pen_pickup` task is designed as a **single continuous episode**. 
*   **Sequence:** Reach → Grasp → Lift → Transport → Drop.
*   **Approach:** Instead of splitting stages into separate tasks, a multi-phase **shaped reward function** guides the agent through the sequence. This is preferred for generating continuous VLA (Vision-Language-Action) data and RL training.

---

## 2. Command Reference (CLI)

All commands use `uv run --active`. (for enviroment that already installed isaac sim and lab on it)

| Action | Command |
| :--- | :--- |
| **Test (Random Agent)** | `uv run --active random_agent --task Isaac-SO-ARM101-PenPickup-Play-v0 --enable_cameras` |
| **Train (Fastest/Headless)** | `uv run --active train --task Isaac-SO-ARM101-PenPickup-v0 --headless --num_envs 256` |
| **Train (GUI/Visual)** | `uv run --active train --task Isaac-SO-ARM101-PenPickup-v0 --num_envs 16` |
| **Resume Training** | `uv run --active train --task Isaac-SO-ARM101-PenPickup-v0 --headless --resume` |
| **Play Policy** | `uv run --active play --task Isaac-SO-ARM101-PenPickup-Play-v0 --num_envs 1 --enable_cameras` |
| **Record Video** | `uv run --active play --task Isaac-SO-ARM101-PenPickup-Play-v0 --enable_cameras --video --video_length 500` |

> **Note on Cameras:** Use `--enable_cameras` only for visual debugging or data collection. For RL training, cameras add significant overhead. If cameras are defined in your scene, you must either use the flag or remove them from the config to avoid errors.

---

## 3. Reward Manager (Shaping Terms)

| Term | Weight | Logic | Condition |
| :--- | :--- | :--- | :--- |
| `reaching_pen` | 1.0 | Gaussian (std=5cm) | High when EE is near pen. Full reward when EE is on top of pen. |
| `lifting_pen` | 15.0 | Binary (+15) | Active when pen is >2.5cm above table. |
| `pen_to_holder_coarse`| 16.0 | Gaussian (std=30cm)| Pen <--> Holder dist.Active only when pen is lifted. |
| `pen_to_holder_fine` | 5.0 | Gaussian (std=5cm) | Tight targeting bonus near holder. |
| `pen_above_holder` | 20.0 | Binary (+20) | Pen within 4cm XY of holder & >3cm high. |
| `action_rate` | -0.0001| Penalty | Penalizes jerky/abrupt movements. |
| `joint_vel` | -0.0001| Penalty | Penalizes excessively fast joint movement. |

---

## 4. Configuration & Hyperparameters

### CLI Flags
*Override without editing files*

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--num_envs N` | `1` | Number of parallel environments (16–64 is a practical range on a single RTX 6000 Ada. More envs = more data per iteration = faster convergence, but more VRAM) |
| `--max_iterations N` | `5000` | Total training iterations |
| `--seed N` | `42` | Random seed for reproducibility |
| `--device cuda:0` | `cuda:0` | GPU to use |
| `--log_interval N` | `1` | Print stats every N iterations |

### PPO Agent Configuration
*Located in `agents/rsl_rl_ppo_cfg.py`*

| Parameter | Current | Description |
| :--- | :--- | :--- |
| `num_steps_per_env` | `24` | Rollout length before update (higher = more stable gradients / better credit assignment in long tasks) |
| `max_iterations` | `5000` | Total training budget |
| `save_interval` | `50` | Checkpoint frequency |
| `learning_rate` | `1e-4` | Step size (lower = more stable, slower) |
| `schedule` | `adaptive` | `adaptive` (hits desired KL) or `fixed` |
| `num_learning_epochs` | `5` | Passes over each rollout batch |
| `num_mini_batches` | `4` | Number of mini-batches per rollout |
| `gamma` | `0.98` | Discount factor (~50 steps look-ahead factor) |
| `entropy_coef` | `0.006` | Exploration bonus (higher = more randomness) |
| `clip_param` | `0.2` | PPO clipping to prevent too-large policy updates |
| `init_noise_std` | `1.0` | Initial policy randomness |
| `actor_hidden_dims` | `[256,128,64]` | Neural network architecture |

### Environment Configuration
*Located in `pen_pickup_env_cfg.py`*

| Parameter | Location | Effect |
| :--- | :--- | :--- |
| `episode_length_s` | `__post_init__` | Max episode length in seconds |
| `env_spacing` | `scene` line | Distance between parallel environments. Default `2.5`m. Can be reduced to `2.0`m to save VRAM |
| `REACH_MIN/MAX` | constants | Spawn region size |
| `Reward weights` | `RewardsCfg` | Relative importance of each phase |

---

## 5. Monitoring & Logs

### Terminal Stats (Training Health)
Watch these columns to ensure the agent is learning:
*   **Rew**: Mean reward. Should trend upward.
*   **EpLen**: Mean episode length. Should increase early on as the agent "survives" longer.
*   **VFnc**: Value function loss. Should decrease as the critic improves.
*   **MeanStd**: Policy noise. Should decrease toward `~0.1` as the policy converges.

### Sign of Learning
*   **Rew**: Increasing over iterations is the primary indicator of learning.
*   **EpLen**: Should increase from low values (early terminations) to higher values as the agent learns to avoid early failures.
*   **holder_knocked**: Rate drops as the agent learns to keep the holder upright.
*   **pen_in_holder > 0**: Eventually, may take a while

### Signs something is wrong
*   **Rew**: Flat or negative with no upward trend after 300+ iterations --> rewards weight need tuning
*   **EpLen**: Stays low (e.g. <5s) --> likely a termination condition is firing too early (e.g. "pen dropped" terminations)
*   **holder_knocked near 1.0**: tilt axis, re-check termination logic and thresholds

### Checkpoint Locations
Checkpoints and configs are saved to:
`logs/rsl_rl/pen_pickup/<timestamp>/`
*   `model_N.pt`: Checkpoint files.
*   `params/env.yaml`: Snapshot of the environment config.
*   `exported/`: Contains `policy.pt` (TorchScript) and `policy.onnx` after running the play script.

```
logs/
└── rsl_rl/
    └── pen_pickup/
        └── 2026-03-06_12-00-00/        ← timestamp of run
            ├── model_50.pt             ← checkpoint every 50 iters
            ├── model_100.pt
            ├── ...
            ├── model_5000.pt           ← final
            └── params/
                ├── env.yaml            ← full env config snapshot
                └── agent.yaml          ← full PPO config snapshot
```

When you run the play script pointing at a checkpoint, it auto-exports to:

```
logs/rsl_rl/pen_pickup/<timestamp>/exported/
├── policy.pt    ← TorchScript (for PyTorch deployment)
└── policy.onnx  ← ONNX (for GR00T / cross-framework)
```
---

## 6. Recommended Workflow

1.  **Physics Check:** Run `add_physics.py` once to ensure USD files are dynamic (not kinematic).
2.  **Visual Sanity:** Run `random_agent --enable_cameras` to ensure the pen spawns correctly and the holder is upright.
3.  **Initial Train:** Run `train --num_envs 16`. Watch the **Rew** trend for 100–200 iterations.
4.  **Visualize Curves:** Run `tensorboard --logdir logs/rsl_rl` in a separate terminal.
5.  **Evaluate:** After ~1000 iterations, run the `play` script to see if the agent is actually attempting to lift the pen.
6.  **Tune:** If the agent never lifts the pen, increase `lifting_pen` weight or temporarily disable "pen dropped" terminations.

---

## 7. Debugging
*   **Is it learning?** If Reward is flat for 300+ iterations, your reward weights are likely off, or a termination is firing too early.
*   **Stuck at Reach?** Reach→Grasp is the hardest transition. Consider adding a reward for gripper width (closing) when the EE is within 1cm of the pen.
*   **Sim-to-Real:** Ensure `randomize_yaw=True` for the pen. This creates a more robust policy for real-world deployment (GR00T pipeline).


---

## 8. Note to self / Plan of action:

### Training:

Start with fewer envs (16) to confirm there are no crashes, then scale to 64-128 once stable
num_steps_per_env=24 is short — for a task this long (10s episode) consider increasing to 48 or 64 for better credit assignment
If the agent never lifts the pen after 500 iterations, temporarily set pen_dropped and holder_knocked terminations to disabled (comment them out) so it has more time to explore

### Reward shaping:

The hardest transition to learn is reach→grasp — consider adding a shaped grasping reward based on gripper width when near the pen
pen_above_holder weight of 20 is very high relative to the others — this is intentional to strongly pull the policy toward the final goal, but if it causes instability lower it to 10

### Sim-to-real (GR00T pipeline):

Once the RL policy achieves >5% success rate in sim, the trajectories from play are useful for GR00T fine-tuning even if they're not perfect
The random pen orientation (randomize_yaw=True) you added is important — it means your synthetic trajectories cover the full range of approach angles, which transfers better to real