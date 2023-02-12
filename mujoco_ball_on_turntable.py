# Copyright 2020 The dm_control Authors and Copyright 2023 Randall Britten
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0.
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.


# Based on https://github.com/deepind/dm_control/blob/main/tutorial.ipynb ("the original") (commit 774f4)
# Changes relative to the original
# - Extracted just python code from notebook, and adapted for being run in a script rather than notebook
# - Extracted just code that pertained to "tippe-top", and adapted to "ball on turntable"
# - Changed some parameters (e.g. resolution, simulation time, framerate)

from dm_control import mujoco

import numpy as np

# Graphics-related
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt

# Font sizes
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def save_video(frames, framerate=30, playback_speed=1.0):
    height, width, _ = frames[0].shape
    dpi = 400
    orig_backend = matplotlib.get_backend()
    matplotlib.use('Agg')  # Switch to headless 'Agg' to inhibit figure rendering.
    fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
    matplotlib.use(orig_backend)  # Switch back to the original backend.
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.set_position([0, 0, 1, 1])
    im = ax.imshow(frames[0])
    def update(frame):
      im.set_data(frame)
      return [im]
    interval = 1000/(framerate * playback_speed)
    anim = animation.FuncAnimation(fig=fig, func=update, frames=frames,
                                   interval=interval, blit=True, repeat=False)
    FFwriter = animation.FFMpegWriter(fps=framerate * playback_speed)
    # anim.save('animation.gif') # Uncomment to save as GIF.
    anim.save('mujoco_ball_on_turntable.mp4', writer = FFwriter) # Assumes ffmpeg is installed. (Hint: use conda install ffmpeg)


physics = mujoco.Physics.from_xml_path("mujoco_ball_on_turntable.xml")

duration = 9   # (seconds)
framerate = 60  # (Hz)

# For video.
frames = []

# For values
timevals = []
angular_velocity = []
ball_x = []
ball_y = []
ball_xyz = []

physics.reset(0)  # Reset to keyframe 0 (load a saved state).
while physics.data.time < duration:
  physics.step()

  if len(frames) < (physics.data.time) * framerate:
    pixels = physics.render(camera_id='closeup')
    frames.append(pixels)
  
  timevals.append(physics.data.time)
  angular_velocity.append(physics.data.qvel[3:6].copy())
  ball_x.append(physics.named.data.geom_xpos['ball1_geom', 'x'])
  ball_y.append(physics.named.data.geom_xpos['ball1_geom', 'y'])
  ball_xyz.append(physics.data.qpos[0:3].copy())

  # capture x, y of ball and angular velocity of ball every 0.5 seconds
  if physics.data.time % 0.5 < physics.model.opt.timestep:
    print(f'{" ".join(map(str, physics.data.qpos[0:2]))}  2.11 {" ".join(map(str, physics.data.qpos[3:7]))} ')

# playback_speed = 1.0
# save_video(frames, framerate, playback_speed)

# dpi = 100
# width = 480
# height = 980
# figsize = (width / dpi, height / dpi)
# _, ax = plt.subplots(3, 1, figsize=figsize, dpi=dpi, sharex=False)
# # space subplots so that title doesn't overlap with x-axis labels
# plt.subplots_adjust(hspace=0.5)

# ax[0].plot(timevals, angular_velocity)
# ax[0].set_xlabel('time(seconds)')
# ax[0].set_ylabel('radians / second')
# ax[0].set_title('Ball angular velocity')

# ax[1].plot(ball_x, ball_y)
# ax[1].set_xlabel('ball x (metres)')
# ax[1].set_ylabel('ball y (meters)')
# ax[1].set_title('Ball path')

# ax[2].plot(timevals, ball_xyz)
# ax[2].set_xlabel('time(seconds)')
# ax[2].set_ylabel('ball coordinates (metres)')
# ax[2].set_title('Ball coordinates')


# plt.savefig("mujoco_ball_on_turntable_plots.png")
