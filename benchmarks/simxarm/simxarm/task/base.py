import os

import numpy as np
import glfw
import gym
from gym.envs.robotics import robot_env

from simxarm.task import mocap


class Base(robot_env.RobotEnv):
	"""
	Superclass for all simxarm environments.
	Args:
		xml_name (str): name of the xml environment file
		gripper_rotation (list): initial rotation of the gripper (given as a quaternion)
	"""
	def __init__(self, xml_name, gripper_rotation=[0,1,0,0]):
		self.gripper_rotation = np.array(gripper_rotation, dtype=np.float32)
		self.center_of_table = np.array([1.655, 0.3, 0.63625])
		self.max_z = 1.2
		self.min_z = 0.2
		super().__init__(
			model_path=os.path.join(os.path.dirname(__file__), 'assets', xml_name + '.xml'),
			n_substeps=20, n_actions=4, initial_qpos={}
		)

	@property
	def dt(self):
		return self.sim.nsubsteps * self.sim.model.opt.timestep

	@property
	def eef(self):
		return self.sim.data.get_site_xpos('grasp')

	@property
	def obj(self):
		return self.sim.data.get_site_xpos('object_site')

	@property
	def robot_state(self):
		gripper_angle = self.sim.data.get_joint_qpos('right_outer_knuckle_joint')
		return np.concatenate([self.eef, [gripper_angle]])

	def is_success(self):
		return NotImplementedError()
	
	def get_reward(self):
		raise NotImplementedError()
	
	def _sample_goal(self):
		raise NotImplementedError()

	def get_obs(self):
		return self._get_obs()

	def _step_callback(self):
		self.sim.forward()

	def _limit_gripper(self, gripper_pos, pos_ctrl):
		if gripper_pos[0] > self.center_of_table[0] -0.105 + 0.15:
			pos_ctrl[0] = min(pos_ctrl[0], 0)
		if gripper_pos[0] < self.center_of_table[0] -0.105 - 0.3:
			pos_ctrl[0] = max(pos_ctrl[0], 0)
		if gripper_pos[1] > self.center_of_table[1] + 0.3:
			pos_ctrl[1] = min(pos_ctrl[1], 0)
		if gripper_pos[1] < self.center_of_table[1] - 0.3:
			pos_ctrl[1] = max(pos_ctrl[1], 0)
		if gripper_pos[2] > self.max_z:
			pos_ctrl[2] = min(pos_ctrl[2], 0)
		if gripper_pos[2] < self.min_z:
			pos_ctrl[2] = max(pos_ctrl[2], 0)
		return pos_ctrl

	def _apply_action(self, action):
		assert action.shape == (4,)
		action = action.copy()
		pos_ctrl, gripper_ctrl = action[:3], action[3]
		pos_ctrl = self._limit_gripper(self.sim.data.get_site_xpos('grasp'), pos_ctrl) * (1/self.sim.nsubsteps)
		gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
		mocap.apply_action(self.sim, np.concatenate([pos_ctrl, self.gripper_rotation, gripper_ctrl]))

	def _viewer_setup(self):
		body_id = self.sim.model.body_name2id('link7')
		lookat = self.sim.data.body_xpos[body_id]
		for idx, value in enumerate(lookat):
			self.viewer.cam.lookat[idx] = value
		self.viewer.cam.distance = 4.0
		self.viewer.cam.azimuth = 132.
		self.viewer.cam.elevation = -14.

	def _render_callback(self):
		self.sim.forward()

	def _reset_sim(self):
		self.sim.set_state(self.initial_state)
		self._sample_goal()
		for _ in range(10):
			self.sim.step()
		return True

	def _set_gripper(self, gripper_pos, gripper_rotation):
		self.sim.data.set_mocap_pos('robot0:mocap2', gripper_pos)
		self.sim.data.set_mocap_quat('robot0:mocap2', gripper_rotation)
		self.sim.data.set_joint_qpos('right_outer_knuckle_joint', 0)
		self.sim.data.qpos[10] = 0.0
		self.sim.data.qpos[12] = 0.0

	def _env_setup(self, initial_qpos):
		for name, value in initial_qpos.items():
			self.sim.data.set_joint_qpos(name, value)
		mocap.reset(self.sim)
		self.sim.forward()
		self._sample_goal()
		self.sim.forward()

	def reset(self):
		self._reset_sim()
		return self._get_obs()

	def step(self, action):
		assert action.shape == (4,)
		assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
		self._apply_action(action)
		for _ in range(2):
			self.sim.step()
		self._step_callback()
		obs = self._get_obs()
		reward = self.get_reward()
		done = False
		info = {'is_success': self.is_success(), 'success': self.is_success()}
		return obs, reward, done, info

	def render(self, mode='rgb_array', width=384, height=384):
		self._render_callback()
		if mode == 'rgb_array':
			return self.sim.render(width, height, camera_name='camera0', depth=False)[::-1, :, :]
		elif mode == "human":
			self._get_viewer(mode).render()

	def close(self):
		if self.viewer is not None:
			# self.viewer.finish()
			print("Closing window glfw")
			glfw.destroy_window(self.viewer.window)
			self.viewer = None
		self._viewers = {}
