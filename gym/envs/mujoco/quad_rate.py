# **********************************************************************
#
# Copyright (c) 2019, Autonomous Systems Lab
# Author: Inkyu Sa <enddl22@gmail.com>
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
# 3. Neither the name PX4 nor the names of its contributors may be
#    used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
# AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# *************************************************************************

import numpy as np
from numpy import linalg
from gym import utils
import os
from gym.envs.mujoco import mujoco_env
import math

# For testing whether a number is close to zero
_FLOAT_EPS = np.finfo(np.float64).eps
_EPS4 = _FLOAT_EPS * 4.0

class QuadRateEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.avg_rwd=-3.0 #obtained from eprewmean
        self.gamma=0.99 #ppo2 default setting value
        self.log_cnt=0
        mujoco_env.MujocoEnv.__init__(self, 'quadrotor_quat_fancy.xml', 5)
        utils.EzPickle.__init__(self)
    def step(self, action):
        mass=self.get_mass()
        act_min=[3.5,-0.5,-0.7,-0.03]
        act_max=[30,0.5,0.7,0.03]
        action = np.clip(action, a_min=act_min, a_max=act_max)
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        pos = ob[0:3]
        quat = ob[3:7]
        lin_vel = ob[7:10]
        ang_vel = ob[10:13]
        reward_ctrl = - 1e-4 * np.sum(np.square(action))
        reward_position = -linalg.norm(pos) * 1e-1
        reward_linear_velocity = -linalg.norm(lin_vel) * 1e-2
        reward_angular_velocity = -linalg.norm(ang_vel) * 1e-3
        reward_alive = 1e-1
        reward = reward_ctrl+reward_position+reward_linear_velocity+reward_angular_velocity+reward_alive
        done= abs(pos[2]) >50 \
                or abs(pos[0]) > 50.0 \
                or abs(pos[1]) > 50.0
        info = {
            'rwp': reward_position,
            'rwlv': reward_linear_velocity,
            'rwav': reward_angular_velocity,
            'rwctrl': reward_ctrl,
            'obx': pos[0],
            'oby': pos[1],
            'obz': pos[2],
            'obvx': lin_vel[0],
            'obvy': lin_vel[1],
            'obvz': lin_vel[2],
        }
        if done:
        	reward = self.avg_rwd / (1-self.gamma)*2#-13599.99
        if (self.log_cnt==1e4):
             print("x={},y={},z={}\n".format(pos[0],pos[1],pos[2]))
             print("thrust={}, dx={}, dy={}, dz={}".format(action[0],action[1],action[2],action[3]))
             self.log_cnt=0
        else: self.log_cnt=self.log_cnt+1
        return ob, reward, done, info

    def _get_obs(self):
        pos = self.sim.data.qpos*1e-0
        vel = self.sim.data.qvel*1e-0
        return np.concatenate([pos.flat,vel.flat])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.1, high=0.1)
        qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.05, high=0.05)
        self.set_state(qpos, qvel)
        observation = self._get_obs();
        return observation

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 1
        v.cam.distance = self.model.stat.extent * 4
        v.cam.azimuth = 132.
        #v.cam.lookat[2] += .8
        #v.cam.elevation = 0
        #v.cam.lookat[0] += 1.5
        v.cam.elevation +=0.9
    def get_mass(self):
        mass = np.expand_dims(self.model.body_mass, axis=1)
        return mass

    #stealed from rotations.py
    def quat2mat(self,quat):
        """ Convert Quaternion to Rotation matrix.  See rotation.py for notes """
        quat = np.asarray(quat, dtype=np.float64)
        assert quat.shape[-1] == 4, "Invalid shape quat {}".format(quat)

        w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
        Nq = np.sum(quat * quat, axis=-1)
        s = 2.0 / Nq
        X, Y, Z = x * s, y * s, z * s
        wX, wY, wZ = w * X, w * Y, w * Z
        xX, xY, xZ = x * X, x * Y, x * Z
        yY, yZ, zZ = y * Y, y * Z, z * Z

        mat = np.empty(quat.shape[:-1] + (3, 3), dtype=np.float64)
        mat[..., 0, 0] = 1.0 - (yY + zZ)
        mat[..., 0, 1] = xY - wZ
        mat[..., 0, 2] = xZ + wY
        mat[..., 1, 0] = xY + wZ
        mat[..., 1, 1] = 1.0 - (xX + zZ)
        mat[..., 1, 2] = yZ - wX
        mat[..., 2, 0] = xZ - wY
        mat[..., 2, 1] = yZ + wX
        mat[..., 2, 2] = 1.0 - (xX + yY)
        return np.where((Nq > _FLOAT_EPS)[..., np.newaxis, np.newaxis], mat, np.eye(3))

    def RotToRPY(self,R):
        R=R.reshape(3,3) #to remove the last dimension i.e., 3,3,1
        phi = math.asin(R[1,2])
        psi = math.atan2(-R[1,0]/math.cos(phi),R[1,1]/math.cos(phi))
        theta = math.atan2(-R[0,2]/math.cos(phi),R[2,2]/math.cos(phi))
        return phi,theta,psi