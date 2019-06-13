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
import mujoco_py

# For testing whether a number is close to zero
_FLOAT_EPS = np.finfo(np.float64).eps
_EPS4 = _FLOAT_EPS * 4.0

class BallBouncingQuadEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.avg_rwd=-3.0 #obtained from eprewmean
        self.gamma=0.99 #ppo2 default setting value
        self.log_cnt=0
        self.z_offset=0.3 #bouncing the ball 30 cm above quad
        self.hit_cnt=0
        self.ball_id=None
        self.quad_id=None
        self.quad_hit_floor=False
        mujoco_env.MujocoEnv.__init__(self, 'ball_bouncing_quad_fancy.xml', 5)
        utils.EzPickle.__init__(self)
        self.ball_id=self.sim.model.geom_name2id('ball')
        self.quad_id=self.sim.model.geom_name2id('core')
    def step(self, action):
        mass=self.get_mass()
        act_min=[3.5,-1.7,-1.7,-0.1]
        act_max=[45,1.7,1.7,0.1]
        action = np.clip(action, a_min=act_min, a_max=act_max)
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        quad_pos = ob[0:3]
        quad_quat = ob[3:7]
        ball_pos = ob[7:10]
        quad_lin_vel = ob[14:17]
        quad_ang_vel = ob[17:20]
        ball_vel = ob[20:23]
        self.checkContact()
        reward_ctrl = - 1e-4 * np.sum(np.square(action))
        reward_position = - linalg.norm(quad_pos[0:2]-ball_pos[0:2])* 1e-1
        reward_quad_z_position = -linalg.norm(quad_pos[2]) * 1e-1
        if (self.hit_cnt>0):
            reward_bouncing_bonus = 5e-1
        else: reward_bouncing_bonus = 0
        reward_alive = 1e-1
        reward = reward_ctrl+reward_position+reward_bouncing_bonus \
                +reward_alive+reward_quad_z_position\
                +reward_alive
        done= abs(quad_pos[2]) >50 \
                or abs(quad_pos[0]) > 50.0 \
                or abs(quad_pos[1]) > 50.0 \
                or ball_pos[2] <= quad_pos[2] -0.5\
                or self.quad_hit_floor
        info = {
            'rwp': reward_position,
            'rwctrl': reward_ctrl,
            'obxq': quad_pos[0],
            'obyq': quad_pos[1],
            'obzq': quad_pos[2],
            'obxb': ball_pos[0],
            'obyb': ball_pos[1],
            'obzb': ball_pos[2],
        }
        if done:
            reward = -100
        if (self.log_cnt==1e4):
             print("x={},y={},z={}\n".format(quad_pos[0]-ball_pos[0],quad_pos[1]-ball_pos[1],quad_pos[2]-ball_pos[2]))
             self.log_cnt=0
        else: self.log_cnt=self.log_cnt+1
        return ob, reward, done, info
    def print_contact_info(self):
        for i in range(self.sim.data.ncon):
            # Note that the contact array has more than `ncon` entries,
            # so be careful to only read the valid entries.
            contact = self.sim.data.contact[i]
            geom2_body = self.sim.model.geom_bodyid[self.sim.data.contact[i].geom2]
            c_array = np.zeros(6, dtype=np.float64)
            mujoco_py.functions.mj_contactForce(self.sim.model, self.sim.data, i, c_array)
            print('c_array', c_array)
            if c_array[0]>8 and c_array[1] < 1e-3 and c_array[2] < 1e-3:
                print("============ball collided=======")
                self.hit_cnt+=1

    def checkContact(self):
        if self.sim.data.ncon>0:
            #print('number of contacts', self.sim.data.ncon)
            for i in range(self.sim.data.ncon):
                contact = self.sim.data.contact[i]
                geom1_name=self.sim.model.geom_id2name(contact.geom1)
                geom2_name=self.sim.model.geom_id2name(contact.geom2)
                if (geom1_name=='core' and geom2_name=='ball') or (geom1_name=='ball' and geom2_name=='core'):
                    c_array = np.zeros(6, dtype=np.float64)
                    mujoco_py.functions.mj_contactForce(self.sim.model, self.sim.data, i, c_array)
                    if c_array[0]>4 and c_array[1] < 3e-3 and c_array[2] < 3e-3:
                        self.hit_cnt+=1
                elif (geom1_name=='core' and geom2_name=='floor') or (geom1_name=='floor' and geom2_name=='core'):
                    self.quad_hit_floor=True


    def _get_obs(self):
        pos = self.sim.data.qpos
        vel = self.sim.data.qvel
        return np.concatenate([pos.flat,vel.flat])

    def reset_model(self):
        self.hit_cnt=0
        self.quad_hit_floor=False
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.05, high=0.05)
        qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        self.set_state(qpos, qvel)
        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent * 4
        v._run_speed=0.1#0.01#0.1 #1
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