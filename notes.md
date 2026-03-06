pen_pickup teaches the entire sequence as one task: reach pen → grasp → lift → transport to holder → drop. The reward function is multi-phase (shaped rewards that guide the policy through each stage), but from the agent's perspective it's a single continuous episode. This is the correct approach for your use case. Splitting into separate tasks would require a task-sequencing layer and doesn't work well for RL training or for generating VLA data (you want continuous trajectories).

========================


Dummy

uv run --active random_agent --task Isaac-SO-ARM101-PenPickup-Play-v0 --enable_cameras


Train (headless, no GUI — fastest)

uv run --active train --task Isaac-SO-ARM101-PenPickup-v0 --headless --num_envs 256 --enable_cameras


Note: --enable_cameras is needed because cameras are defined in the scene. Without it, camera sensors will error. If you don't need camera data during RL training (just joint-state policy), you could remove the cameras from the scene config for faster training and add them back for data generation.

Might need to check if the camera actually records the image they are seeing



Train (with GUI — watch it learn)

uv run --active train --task Isaac-SO-ARM101-PenPickup-v0 --num_envs 16 --enable_cameras

Resume training from last checkpoint

uv run --active train --task Isaac-SO-ARM101-PenPickup-v0 --headless --num_envs 256 --enable_cameras --resume

Play a trained policy

uv run --active play --task Isaac-SO-ARM101-PenPickup-Play-v0 --enable_cameras --num_envs 1

Play and record video

uv run --active play --task Isaac-SO-ARM101-PenPickup-Play-v0 --enable_cameras --num_envs 1 --video --video_length 500



======


Reward Manager (7 terms):

Term	Weight	Meaning
reaching_pen	1.0	Gaussian on EE↔pen distance, std=5cm. Full reward when EE is on top of pen
lifting_pen	15.0	Binary: +15 every step when pen is >2.5cm above table
pen_to_holder_coarse	16.0	Gaussian std=30cm on pen↔holder dist. Active only when pen is lifted
pen_to_holder_fine	5.0	Same but std=5cm — tight targeting bonus
pen_above_holder	20.0	Binary: +20 when pen is within 4cm XY of holder AND >3cm above table
action_rate	-0.0001	Penalty for jerky motion (change in action from step to step)
joint_vel	-0.0001	Penalty for fast joints



Keyboard shortcuts during simulation
Key	Action
V	Toggle viewport camera between perspective views
F	Frame selected object in viewport
Space	Pause/unpause simulation
Right arrow	Step one frame (when paused)
Escape	Stop simulation and exit
Mouse right-drag	Rotate viewport camera
Mouse middle-drag	Pan
Scroll	Zoom



Training health — in the training terminal watch these columns:

Rew       — mean episode reward, should increase over time
EpLen     — mean episode length, should increase early (agent survives longer)  
VFnc      — value function loss, should decrease
MeanStd   — policy noise std, should decrease as policy converges (~0.1 is good)

If Rew stays flat for 200+ iterations, reward weights need tuning or there's a physics issue.