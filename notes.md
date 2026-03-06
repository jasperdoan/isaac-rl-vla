uv run --active random_agent --task Isaac-SO-ARM101-PenPickup-Play-v0 --enable_cameras



So first of all nice job. A few minor changes:

One we need to run `uv run --active random_agent --task Isaac-SO-ARM101-PenPickup-Play-v0 --enable_cameras` to enable cameras

two, the wrist camera should move along with the arm, currently it is just fixed in place, the angle currently is also point wrong, I want it to point to where the claw is, so that we can see the pen in the wrist camera when the arm is picking it up. So maybe change the angle, and tell me where to change it so I can tweak it around if needed.

The table should be wider and longer. Can you tell me where to change the dimensions of the table also so that I can just manually tweak it around if needed? 

The pen and 