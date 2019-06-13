## Notes
This is a forked repository of [OpenAI Gym](https://github.com/openai/gym), so that functionalities and environemtns are identical except we add two quadrotor models, [rate control](https://github.com/inkyusa/gym_rotor/blob/master/gym/envs/mujoco/quad_rate.py) and [ball bounding quadrotor model](https://github.com/inkyusa/gym_rotor/blob/master/gym/envs/mujoco/ball_bouncing_quad.py). Threfore, this repo only focuses on these models and please refer to [OpenAI Gym](https://github.com/openai/gym) for more detail (e.g., installation, and other models).


## Training and testing system requirements
We use [Mujoco 2.0](http://www.mujoco.org/) and [Mujoco-py](https://github.com/openai/mujoco-py) (python wrapper). It is recommended to create a python 3.x [conda](https://docs.conda.io/en/latest/miniconda.html) environment for simplicity.
Once you install this repo, check installed environments by executing the following
```shell
$python
from gym import envs
print(envs.registry.all())
```
You should be able to find two custom environments; “QuadRate-v0” and “BallBouncingQuad-v0”.
You also need the following repos for testing and training these environments.
* https://github.com/inkyusa/rl-baselines-zoo
* https://github.com/inkyusa/stable-baselines (installation required)
* https://github.com/inkyusa/gym_rotor (installation required)
* https://github.com/inkyusa/openai_train_scripts


## 1. Continuous control a quadrotor via rate commands (rate control)
The animation below is what you can expect after training your rate control model. Rate control implies that we command `body-rate` for 3 axes; roll (rotating along x-axis which is forward direction of the vehicle), pitch (rotating along y-axis which is left), and yaw (rotating along z-axis which is up); <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;\boldsymbol{u}=[u_{\dot{x}},u_{\dot{y}},u_{\dot{z}},u_{T}]^{T}" target="_blank"><img src="https://latex.codecogs.com/png.latex?\large&space;\boldsymbol{u}=[u_{\dot{x}},u_{\dot{y}},u_{\dot{z}},u_{T}]^{T}" title="\large \boldsymbol{u}=[u_{\dot{x}},u_{\dot{y}},u_{\dot{z}},u_{T}]^{T}" /></a>. The units are `rad/s` for rates and [0,1] for thrust.

The task of this environment is very simple that we provide a goal position, <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;\boldsymbol{p^{*}}=[x,y,z]^{T}" target="_blank"><img src="https://latex.codecogs.com/png.latex?\large&space;\boldsymbol{p^{*}}=[x,y,z]^{T}" title="\large \boldsymbol{p^{*}}=[x,y,z]^{T}" /></a> and a policy is trained to minimize goal to vehicle distance (i.e., maximize cumulative reward). For the detail for reward shaping, please have a look [here](https://github.com/inkyusa/gym_rotor/blob/ac843fe34d6c5e316a0ae2d8143be22e4298864b/gym/envs/mujoco/quad_rate.py#L69). For training, we use PPO2 provided from [stable-baselines](https://github.com/inkyusa/stable-baselines). In summary, we have 4 input commands, <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;\boldsymbol{u}" target="_blank"><img src="https://latex.codecogs.com/png.latex?\large&space;\boldsymbol{u}" title="\large \boldsymbol{u}" /></a>, and 13 observations, <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;\boldsymbol{o}=[\bold{p},\bold{q},\bold{v},\bold{\omega}]^{T}" target="_blank"><img src="https://latex.codecogs.com/png.latex?\large&space;\boldsymbol{o}=[\bold{p},\bold{q},\bold{v},\bold{\omega}]^{T}" title="\large \boldsymbol{o}=[\bold{p},\bold{q},\bold{v},\bold{\omega}]^{T}" /></a>. Note that <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;\bold{q}" target="_blank"><img src="https://latex.codecogs.com/png.latex?\large&space;\bold{q}" title="\large \bold{q}" /></a> is unit quaternion (4x1) and others are correspondence to position, linear-, angular-velocity respectively (3x1).

In this example, we set  <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;\boldsymbol{p^{*}}=[0,0,1]^{T}" target="_blank"><img src="https://latex.codecogs.com/png.latex?\large&space;\boldsymbol{p^{*}}=[0,0,1]^{T}" title="\large \boldsymbol{p^{*}}=[0,0,1]^{T}" /></a>.

<p align="center"> <img src="http://drive.google.com/uc?export=view&id=17X2N80lA2Ciq8fC9GTyJVR6G-x4Bjy9t" width="550" /> </p>

[quad_rate.py](https://github.com/inkyusa/gym_rotor/blob/master/gym/envs/mujoco/quad_rate.py) is OpenAI environment file and [quadrotor_quat.xml](https://github.com/inkyusa/gym_rotor/blob/master/gym/envs/mujoco/assets/quadrotor_quat.xml) is Mujoco model file that describes physical vehicle model and other simulation properties such as air-density, gravity, viscosity, and so on. [quadrotor_quat_fancy.xml](https://github.com/inkyusa/gym_rotor/blob/master/gym/envs/mujoco/assets/quadrotor_quat_fancy.xml) is only for fancier rendering (more effects for lighting, shadow etc.) which usually takes more time to visualize. It is thus recommdended to use [quadrotor_quat.xml](https://github.com/inkyusa/gym_rotor/blob/master/gym/envs/mujoco/assets/quadrotor_quat.xml) for the sake of training time.

### 1.1 Testing pre-trained weight
We provide a pre-trained weight and you can obtain it from another [repository](https://github.com/inkyusa/openai_train_scripts).
- clone it and go to openai_train_scripts folder and execute the follow command
```shell
source ./rateQuad_test_script.sh ./model
```
You should be able to see the same animation we saw earlier.
#### 1.1.1 What does the policy learn?
As we can see from the above animation, our agent is able to fly to the goal and hover at that position. In order to do this task, the policy has to learn underlying `attitude` and `position` controllers. The former governs to control attitudes of the vehicle which are roll, pitch, and yaw angles and the latter deals with regulating position (i.e., tracking position error and minimiing it). 

### 1.2 Hyperparameters
It is often quite important to properly tune hyperparameters for a particular environment yet PPO2 is relatively robust to these params. We use the following setup as a suboptimal configuration and it seems to work well. But always you are more than welcome to tune your own params and test it.
One can find hyperparameters from [here](https://github.com/inkyusa/rl-baselines-zoo/blob/3173d8fc6c9127446ab639a2413e7d8bbc5eff2a/hyperparams/ppo2.yml#L266)

The table below summarizes hyperparameters used for training both rate control and ball bouncing quadrotor. 

| Name of param | Value         |
| ---           | ---           |
| normalize     | true          |
| n_envs        | 32            |
| n_timesteps   | 50e7          |
| policy        | 'MlpPolicy'   |
| policy_act_fun| 'tanh'        |
| n_steps        | 2048   |
| nminibatches   | 50e7          |
| lam        | 0.95 |
| noptepochs        | 10   |
| ent_coef        | 0.001   |
| learning_rate | 2.5e-4 |
| cliprange | 0.2 |
| max_episode_steps | 8000 |
| reward_threshold | 9600|

### 1.3 Training procedures
Analogous to above testing, training can be easily done if you already installed dependencies. Go to openai_train_scripts folder and execute the follow command
```shell
source ./train_rateQuad_script_module.sh
```

## 2. Ball Bouncing Quadrotor (BBQ)
This environment is minor extension of the privous environment such is rate control. We introduce a ball above the vehicle and shape the reward in the way of hitting the ball at the center of the vehicle. Below animation demonstartes this.

<p align="center">  <img src="http://drive.google.com/uc?export=view&id=1JAVMOKZne7Zxp7ALkqSqmFC9MF607xPd" width="550" /> </p>

One tricky thing for this model was simulating elastic collision (Mujoco 1.5 didn't fully suport this). According to their description regarding Mujoco 2.0, full elastic simulation is supported and a user can set it by specifying negative number in solref (see [here](https://github.com/inkyusa/gym_rotor/blob/ac843fe34d6c5e316a0ae2d8143be22e4298864b/gym/envs/mujoco/assets/ball_bouncing_quad.xml#L39)). For those who want to know in-depth explanation, please refer to [link1](http://www.mujoco.org/forum/index.php?threads/fully-elastic-collisions.3656/) and [link2](http://www.mujoco.org/book/modeling.html#CSolver))

The trained policy performs well (I think) but sometimes it can't handle flowing off ball when bouncing is very small.

[ball_bouncing_quad.py](https://github.com/inkyusa/gym_rotor/blob/master/gym/envs/mujoco/ball_bouncing_quad.py) is OpenAI environment file and [ball_bouncing_quad.xml](https://github.com/inkyusa/gym_rotor/blob/master/gym/envs/mujoco/assets/ball_bouncing_quad.xml) is Mujoco model file that describes physical vehicle and ball models and other simulation properties such as air-density, gravity, viscosity, and so on. [ball_bouncing_quad_fancy.xml](https://github.com/inkyusa/gym_rotor/blob/master/gym/envs/mujoco/assets/ball_bouncing_quad_fancy.xml) is only for fancier rendering (more effects for lighting, shadow etc.) which usually takes more time to visualize. It is thus recommdended to use [ball_bouncing_quad.xml](https://github.com/inkyusa/gym_rotor/blob/master/gym/envs/mujoco/assets/ball_bouncing_quad.xml) for the sake of training time. Note that in this Mujoco model, we set `contype` and `conaffinity` as 0 for the vehicle arms and propellers to avoid possible collisions with ball. Only the top plate has `contype` and `conaffinity` of 1 to enable collision with ball. This may be different to real quadrotor scenario.


### 2.1 Testing pre-trained weight
We provide a pre-trained weight and you can obtain it from another [repository](https://github.com/inkyusa/openai_train_scripts).
- clone it and go to openai_train_scripts folder and execute the follow command
```shell
source ./bbq_test_script.sh ./model
```
You should be able to see the same animation we saw earlier.
### 2.2 Hyperparameters
The same hyperparameters used as of rate control model.

### 2.3 Training procedures
Analogous to above testing, training can be easily done if you already installed dependencies. Go to openai_train_scripts folder and execute the follow command
```shell
source ./train_bbq_script_module.sh
```

## 3. Deploying the trained weight to real-world
WIP...but you can have a look our previous [work](https://arxiv.org/abs/1707.05110) on Control of a Quadrotor with Reinforcement Learning (i.e., outputting direct rotor speed commands instead rate command). Please stay tune and we will update once we have some interesting results.

## Publications
If our work helps your works in an academic/research context, please cite the following publication(s):

* Jemin Hwangbo, Inkyu Sa, Roland Siegwart, Marco Hutter, **"Control of a Quadrotor with Reinforcement Learning"**, 2017, [IEEE Robotics and Automation Letters](https://ieeexplore.ieee.org/document/7961277) or ([arxiv pdf](https://arxiv.org/abs/1707.05110))

```bibtex
@ARTICLE{7961277, 
author={J. {Hwangbo} and I. {Sa} and R. {Siegwart} and M. {Hutter}}, 
journal={IEEE Robotics and Automation Letters}, 
title={Control of a Quadrotor With Reinforcement Learning}, 
year={2017}, 
volume={2}, 
number={4}, 
pages={2096-2103}, 
keywords={aircraft control;helicopters;learning systems;neurocontrollers;stability;step response;quadrotor control;reinforcement learning;neural network;step response;stabilization;Trajectory;Junctions;Learning (artificial intelligence);Computational modeling;Neural networks;Robots;Optimization;Aerial systems: mechanics and control;learning and adaptive systems}, 
doi={10.1109/LRA.2017.2720851}, 
ISSN={2377-3766}, 
month={Oct},}
```
