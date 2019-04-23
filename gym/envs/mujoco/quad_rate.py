import numpy as np
from numpy import linalg
from gym import utils
import os
from gym.envs.mujoco import mujoco_env
import math

#from gym_reinmav.envs.mujoco import MujocoQuadEnv

# For testing whether a number is close to zero
_FLOAT_EPS = np.finfo(np.float64).eps
_EPS4 = _FLOAT_EPS * 4.0


class QuadRateEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        #xml_path = os.path.join(os.path.dirname(__file__), "./assets", 'half_cheetah.xml')
        mujoco_env.MujocoEnv.__init__(self, 'quadrotor_quat.xml', 5)
        utils.EzPickle.__init__(self)
        self.avg_rwd=-3.0 #obtained from eprewmean
        self.gamma=0.99 #ppo2 default setting value

    def step(self, action):
        mass=self.get_mass()
        #print("mass=",mass[1])
        #temp_thrust= 
        #action[0] += mass[1]*9.81 #gravity compensation, 0.4*9.81=3.92

        #act_min=[3.5,-0.5,-0.7,-0.03]
        #act_max=[30,0.5,0.7,0.03]
    #     #action = np.clip(action, a_min=-np.inf, a_max=np.inf)
        #action = np.clip(action, a_min=act_min, a_max=act_max)
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        pos = ob[0:3]
        #R = ob[3:12]
        #lin_vel = ob[12:15]
        #ang_vel= ob[15:18]
        quat = ob[3:7]
        lin_vel = ob[7:10]
        #R=self.quat2mat(quat.transpose())
        #rpy = self.RotToRPY(R)
        #print("rpy(degrees) =",np.rad2deg(rpy))
        # reward_ctrl = - 0.1e-2 * np.sum(np.square(action))
        # reward_position = -linalg.norm(pos) * 1e-1
        # reward_linear_velocity = -linalg.norm(lin_vel) * 0.1e-3
        # reward_angular_velocity = -linalg.norm(ang_vel) * 0.1e-3
        # reward = reward_ctrl+reward_position+reward_linear_velocity+reward_angular_velocity
        reward_ctrl = 0 #- 0.1e-2 * np.sum(np.square(action))
        reward_position = -linalg.norm(pos) * 1e-2
        reward_linear_velocity = 0 #-linalg.norm(lin_vel) * 0.1e-3
        reward_angular_velocity = 0 #-linalg.norm(ang_vel) * 0.1e-3

        reward = reward_position
        
        done= abs(pos[2]) >50 \
                or abs(pos[0]) > 50.0 \
                or abs(pos[1]) > 50.0
        # print("status=",status)
        # print("pos=",pos)
        # info = {
        #     'rwp': reward_position,
        #     'rwlv': reward_linear_velocity,
        #     'rwav': reward_angular_velocity,
        #     'rwctrl': reward_ctrl,
        #     'obx': pos[0],
        #     'oby': pos[1],
        #     'obz': pos[2],
        #     'obvx': lin_vel[0],
        #     'obvy': lin_vel[1],
        #     'obvz': lin_vel[2],
        # }
        info = {
            'rwp': reward_position,
            'rwlv': 0,
            'rwav': 0,
            'rwctrl': 0,
            'obx': pos[0],
            'oby': pos[1],
            'obz': pos[2],
            'obvx': lin_vel[0],
            'obvy': lin_vel[1],
            'obvz': lin_vel[2],
        }

        # retOb= np.concatenate([
        #     pos,R.flat,lin_vel,ang_vel])

        if done:
        	reward = self.avg_rwd / (1-self.gamma)#-13599.99
        	#print("terminated reward=",reward)
        #return retOb, reward, done, info
        return ob, reward, done, info

    def _get_obs(self):
        # pos = self.sim.data.qpos.flat[0:3]
        # quat = self.sim.data.qpos.flat[3:7]
        # linVel = 0.1*self.sim.data.qvel.flat[0:3]
        # angVel = 0.1*self.sim.data.qvel.flat[3:6]
        # R=self.quat2mat(quat.transpose())
        # return np.concatenate([pos,R.flat,linVel,angVel])
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
        ])

    def reset_model(self):
        # pos = self.np_random.uniform(size=3, low=-20, high=20)
        # quat = self.np_random.uniform(size=4, low=-1, high=1)
        # linVel = self.np_random.uniform(size=3, low=-2, high=2)
        # angVel = self.np_random.uniform(size=3, low=-0.5, high=0.5)
        # qpos = np.concatenate([pos,quat])
        # qvel = np.concatenate([linVel,angVel])

        qpos = self.init_qpos 
        qvel = self.init_qvel
        #qpos[0:3] += self.np_random.uniform(low=-.1, high=.1, size=3)
        #qpos = self.init_qpos
        #qpos[0:3] = qpos[0:3]+self.np_random.uniform(size=3, low=-10, high=10)

        #ob[3:12] = self.np_random.uniform(size=9, low=-1, high=1)
        #qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        #qvel[0:3] = self.np_random.uniform(size=3, low=-2, high=2)
        #qvel[3:6] = self.np_random.uniform(size=3, low=-0.5, high=0.5)

        self.set_state(qpos, qvel)
        observation = self._get_obs();
        return observation

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent * 10
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