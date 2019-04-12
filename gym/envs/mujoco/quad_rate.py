import numpy as np
from numpy import linalg
from gym import utils
import os
from gym.envs.mujoco import mujoco_env

#from gym_reinmav.envs.mujoco import MujocoQuadEnv


class QuadRateEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        #xml_path = os.path.join(os.path.dirname(__file__), "./assets", 'half_cheetah.xml')
        mujoco_env.MujocoEnv.__init__(self, 'quadrotor_quat.xml', 5)
        utils.EzPickle.__init__(self)

    def step(self, action):
        #print('pos',self.sim.data.qpos)
        mass=self.get_mass()
        #print("mass=",mass[1])
        action[0] += mass[1]*9.81 #gravity compensation, 0.4*9.81=3.92

        act_min=[3.5,-0.5,-0.7,-0.03]
        act_max=[15,0.5,0.7,0.03]
    #     #action = np.clip(action, a_min=-np.inf, a_max=np.inf)
        action = np.clip(action, a_min=act_min, a_max=act_max)
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        #print("ob=",ob)
        pos = ob[0:3]
        lin_vel = ob[7:10]
        reward_ctrl = - 0.1e-5 * np.sum(np.square(action))
        reward_position = -linalg.norm(pos) * 1e-0
        reward_linear_velocity = -linalg.norm(lin_vel) * 0.1e-3 
        #reward = reward_ctrl + reward_position + reward_linear_velocity
        reward = reward_ctrl + reward_position #reward_linear_velocity
        done = False
        return ob, reward, done, dict(reward_ctrl=reward_ctrl, reward_position=reward_position, reward_linear_velocity=reward_linear_velocity)

    def _get_obs(self):
        return np.concatenate([
            #self.sim.data.qpos.flat[1:],
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent * 10
    def get_mass(self):
        mass = np.expand_dims(self.model.body_mass, axis=1)
        return mass
    # def __init__(self, xml_name="quadrotor_quat.xml"):
    #     super(MujocoQuadQuaternionEnv, self).__init__(xml_name=xml_name)

    # def step(self, action):
    #     goal_pos = np.array([0.0, 0.0, 1.0])
    #     alive_bonus = 1e1
    #     xposbefore = self.sim.data.qpos[0]
    #     self.do_simulation(action, self.frame_skip)
    #     xposafter = self.sim.data.qpos[0]
    #     ob = self._get_obs()
    #     pos = ob[0:3]
    #     quat = ob[3:7] 
    #     lin_vel = ob[7:10]
    #     ang_vel= ob[10:13]
    #     lin_acc = ob[13:16]
    #     ang_acc = ob[16:19]
    #     #print("step a=",a)
        

    #     #reward_position = -linalg.norm(pos-goal_pos) * 0.2e-1 
    #     reward_position = -linalg.norm(pos) * 0.2e-1 
    #     reward_linear_velocity = -linalg.norm(lin_vel) * 1e-3 
    #     reward_angular_velocity = -linalg.norm(ang_vel) * 1e-1
    #     reward_action = -linalg.norm(action)+np.sum(action)*1e-1
    #     reward_alive = alive_bonus

    #     # reward = reward_position \
    #     #          + reward_linear_velocity \
    #     #          + reward_angular_velocity \
    #     #          + reward_action \
    #     #          + reward_alive
    #     reward_ctrl = - 0.1 * np.square(action).sum()
    #     reward_run = (xposafter - xposbefore)/self.dt
    #     #print("r_ctrl=",reward_ctrl)
    #     #print("r_run=",reward_run)
    #     reward = reward_ctrl + reward_run

    #     # notdone = np.isfinite(ob).all() \
    #     #           and pos[2] > 0.3 \
    #     #           and abs(pos[0]) < 2.0 \
    #     #           and abs(pos[1]) < 2.0
    #     notdone = np.isfinite(ob).all() \
    #               and abs(pos[0]) < 2.0 \
    #               and abs(pos[1]) < 2.0

    #     # info = {
    #     #   'rp': reward_position,
    #     #   'rlv': reward_linear_velocity,
    #     #   'rav': reward_angular_velocity,
    #     #   'ra': reward_action,
    #     #   'rlive': reward_alive,
    #     # }
    #     # info = {
    #     #   'rp': reward_position,
    #     #   'rlv': reward_linear_velocity,
    #     #   'rav': reward_ctrl,
    #     #   'ra': reward_action,
    #     #   'rlive': reward_run,
    #     # }
    #     info=dict(reward_run=reward_run, reward_ctrl=reward_ctrl)


    #     #if done=True indicates the episode has terminated and it's time to reset the environment. (For example, perhaps the pole tipped too far, or you lost your last life.) https://gym.openai.com/docs/
    #     #done = not notdone
    #     done = False
    #     return ob, reward, done, info

    # def reset_model(self):
    #     #If reset, then we add some variations to the initial state that will be exploited for the next ep. The low and high bounds empirically set.
    #     qpos=self.init_qpos
    #     qvel=self.init_qvel
    #     qpos[0:3] +=self.np_random.uniform(size=3, low=-0.1, high=0.1)
    #     qvel[0:3] +=self.np_random.uniform(size=3, low=-0.01, high=0.01)
    #     self.set_state(qpos, qvel)
    #     return self._get_obs()

    # def clip_action(self, action):
    #     """
    #     clip action to [0, inf]
    #     :param action:
    #     :return: clipped action
    #     """
    #     act_min=[0,-0.5,-0.5,-0.5]
    #     act_max=[7,0.5,0.5,0.5]
    #     #action = np.clip(action, a_min=-np.inf, a_max=np.inf)
    #     action = np.clip(action, a_min=act_min, a_max=act_max)
    #     return action