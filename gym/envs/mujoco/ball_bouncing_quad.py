import numpy as np
from numpy import linalg
from gym import utils
import os
from gym.envs.mujoco import mujoco_env
import math
import mujoco_py

#from gym_reinmav.envs.mujoco import MujocoQuadEnv

# For testing whether a number is close to zero
_FLOAT_EPS = np.finfo(np.float64).eps
_EPS4 = _FLOAT_EPS * 4.0

class BallBouncingQuadEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        #xml_path = os.path.join(os.path.dirname(__file__), "./assets", 'half_cheetah.xml')
        self.avg_rwd=-3.0 #obtained from eprewmean
        self.gamma=0.99 #ppo2 default setting value
        self.log_cnt=0
        self.z_offset=0.1 #bouncing the ball 30 cm above quad
        self.hit_cnt=0
        mujoco_env.MujocoEnv.__init__(self, 'ball_bouncing_quad.xml', 5)
        utils.EzPickle.__init__(self)
        
    def step(self, action):
        mass=self.get_mass()
        #print("mass=",mass[1])
        #temp_thrust= 
        #action[0] += mass[1]*9.81 #gravity compensation, 0.4*9.81=3.92
        #print("gamma=",self.gamma)
        act_min=[3.5,-0.5,-0.7,-0.03]
        act_max=[30,0.5,0.7,0.03]
        #action = np.clip(action, a_min=-np.inf, a_max=np.inf)
        action = np.clip(action, a_min=act_min, a_max=act_max)
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        quad_pos = ob[0:3]
        quad_quat = ob[3:7]
        ball_pos = ob[7:10]
        #R = ob[3:12]
        #lin_vel = ob[12:15]
        #ang_vel= ob[15:18]
        quad_lin_vel = ob[14:17]
        quad_ang_vel = ob[17:20]
        ball_vel = ob[20:23]
        #R=self.quat2mat(quat.transpose())
        #rpy = self.RotToRPY(R)
        #print("rpy(degrees) =",np.rad2deg(rpy))
        #self.print_contact_info()
        self.checkContact()
        reward_ctrl = - 1e-4 * np.sum(np.square(action))
        reward_position = - linalg.norm(quad_pos[0:2]-ball_pos[0:2])* 1e1
        reward_quad_z_position = -linalg.norm(quad_pos[2]) * 1e-1
        
        reward_linear_velocity = -linalg.norm(quad_lin_vel) * 1e-2
        reward_angular_velocity = -linalg.norm(quad_ang_vel) * 1e-3
        # if (ball_pos[2]-quad_pos[2] > self.z_offset):
        #     reward_bouncing_bonus = 5e-1
        # else: reward_bouncing_bonus = 0
        reward_bouncing_bonus = self.hit_cnt*5e-1
        #reward_z_offset = 1/((ball_pos[2]-quad_pos[2])-self.z_offset)

        reward_alive = 1e-1
        #reward = reward_ctrl+reward_position+reward_linear_velocity+reward_angular_velocity+reward_alive #+reward_z_offset
        reward = reward_ctrl+reward_position+reward_linear_velocity \
                +reward_angular_velocity+reward_alive\
                +reward_quad_z_position #+reward_z_offset
        
        done= abs(quad_pos[2]) >50 \
                or abs(quad_pos[0]) > 50.0 \
                or abs(quad_pos[1]) > 50.0 \
                or ball_pos[2] <= quad_pos[2] -0.5
        #         or ball_pos[2] <= quad_pos[2]
        # done= linalg.norm(quad_pos[0:2]-ball_pos[0:2]) > 0.3 \
                # or ball_pos[2] <= quad_pos[2] -0.5

        # done= abs(pos[2]) >50 \
        #         or abs(pos[0]) > 50.0 \
        #         or abs(pos[1]) > 50.0
        # print("quad pos=",quad_pos)
        # print("ball pos=",ball_pos)
        #done = ball_pos[2] <= quad_pos[2]
        # print("status=",status)
        # print("pos=",pos)
        info = {
            'rwp': reward_position,
            #'rwlv': reward_linear_velocity,
            #'rwav': reward_angular_velocity,
            'rwctrl': reward_ctrl,
            'obxq': quad_pos[0],
            'obyq': quad_pos[1],
            'obzq': quad_pos[2],
            'obxb': ball_pos[0],
            'obyb': ball_pos[1],
            'obzb': ball_pos[2],
        }
        # retOb= np.concatenate([
        #     pos,R.flat,lin_vel,ang_vel])
        if done:
        	#reward = self.avg_rwd / (1-self.gamma)*2#-13599.99
        	reward = -self.avg_rwd / (1-self.gamma)*2
            #print("terminated reward=",reward)
        #return retOb, reward, done, info
        if (self.log_cnt==1e4):
             print("x={},y={},z={}\n".format(quad_pos[0]-ball_pos[0],quad_pos[1]-ball_pos[1],quad_pos[2]-ball_pos[2]))
             #print("thrust={}, dx={}, dy={}, dz={}".format(action[0],action[1],action[2],action[3]))
             self.log_cnt=0
        else: self.log_cnt=self.log_cnt+1

        # act_min=[3.5,-1.5,-1.5,-0.3]
        # act_max=[35,1.5,1.5,0.3]
        # action = np.clip(action, a_min=act_min, a_max=act_max)
        # #action = [3.9, 0, 0, 0]
        # self.do_simulation(action, self.frame_skip)
        # ob = self._get_obs()
        return ob, reward, done, info
    def print_contact_info(self):
        # print('number of contacts', self.sim.data.ncon)
        for i in range(self.sim.data.ncon):
            # Note that the contact array has more than `ncon` entries,
            # so be careful to only read the valid entries.
            contact = self.sim.data.contact[i]
            # print('contact', i)
            # print('dist', contact.dist)
            # print('geom1', contact.geom1, self.sim.model.geom_id2name(contact.geom1))
            # print('geom2', contact.geom2, self.sim.model.geom_id2name(contact.geom2))
            # There's more stuff in the data structure
            # See the mujoco documentation for more info!
            geom2_body = self.sim.model.geom_bodyid[self.sim.data.contact[i].geom2]
            # print(' Contact force on geom2 body', self.sim.data.cfrc_ext[geom2_body])
            # print('norm', np.sqrt(np.sum(np.square(self.sim.data.cfrc_ext[geom2_body]))))
            # Use internal functions to read out mj_contactForce
            c_array = np.zeros(6, dtype=np.float64)
            mujoco_py.functions.mj_contactForce(self.sim.model, self.sim.data, i, c_array)
            print('c_array', c_array)
            if c_array[0]>8 and c_array[1] < 1e-3 and c_array[2] < 1e-3:
                print("============ball collided=======")
                self.hit_cnt+=1

    def checkContact(self):
        # dist = linalg.norm(ball_pos - quad_pos)
        # #print("dist=",dist)
        # if dist<0.13:
        #     self._ball_hit_quad=True
        # else:
        #     self._ball_hit_quad=False
        #     #print("collided")
        # Below doesn't work (i.e., inconsistenly detect ball and core collision)
        # so using distance between core and ball instead. 0.13 was experimentally chosen
        if self.sim.data.ncon>0:
            #print('number of contacts', self.sim.data.ncon)
            for i in range(self.sim.data.ncon):
                contact = self.sim.data.contact[i]
                geom1_name=self.sim.model.geom_id2name(contact.geom1)
                geom2_name=self.sim.model.geom_id2name(contact.geom2)
                #print('geom1', contact.geom1, self.sim.model.geom_id2name(contact.geom1))
                #print('geom2', contact.geom2, self.sim.model.geom_id2name(contact.geom2))
                if (geom1_name=='core' and geom2_name=='ball') or (geom1_name=='ball' and geom2_name=='core'):
                    c_array = np.zeros(6, dtype=np.float64)
                    mujoco_py.functions.mj_contactForce(self.sim.model, self.sim.data, i, c_array)
                    #print('c_array', c_array)
                    if c_array[0]>6 and c_array[1] < 3e-2 and c_array[2] < 3e-2:
                        #print("============ball collided=======")
                        self.hit_cnt+=1
                        #print("self.hit_cnt=",self.hit_cnt)


    def _get_obs(self):
        # pos = self.sim.data.qpos*1e-1
        # vel = self.sim.data.qvel*1e-2
        pos = self.sim.data.qpos
        vel = self.sim.data.qvel
        #print("pos=",pos)
        #print("vel=",vel)
        #del temp_ob[10:14] # orientation of the ball
        #del temp_ob[23:26] # angular velocity of the ball
        #print("temp_ob.shape()=",temp_ob.shape) # 26-7=19 state
        return np.concatenate([pos.flat,vel.flat])

    def reset_model(self):
        self.hit_cnt=0
        # pos = self.np_random.uniform(size=3, low=-20, high=20)
        # quat = self.np_random.uniform(size=4, low=-1, high=1)
        # linVel = self.np_random.uniform(size=3, low=-2, high=2)
        # angVel = self.np_random.uniform(size=3, low=-0.5, high=0.5)
        # qpos = np.concatenate([pos,quat])
        # qvel = np.concatenate([linVel,angVel])
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.05, high=0.05)
        qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        #qpos = self.init_qpos 
        #qvel = self.init_qvel

        #qpos[0:3] += self.np_random.uniform(low=-5, high=5, size=3)
        #qpos = self.init_qpos
        #qpos[0:3] = qpos[0:3]+self.np_random.uniform(size=3, low=-10, high=10)

        #ob[3:12] = self.np_random.uniform(size=9, low=-1, high=1)
        #qvel += self.np_random.uniform(size=6, low=-0.5, high=0.5)
        #qvel[0:3] = self.np_random.uniform(size=3, low=-2, high=2)
        #qvel[3:6] = self.np_random.uniform(size=3, low=-0.5, high=0.5)

        self.set_state(qpos, qvel)
        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent * 4
        v._run_speed=0.05#0.1 #1
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
