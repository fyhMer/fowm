from collections import deque, defaultdict
from typing import Any, NamedTuple
import dm_env
import numpy as np
import torch
from dm_env import StepType, specs
import gym
import warnings
import yaml

warnings.filterwarnings("ignore", category=DeprecationWarning)

"""Wrappers based on drqv2 (https://github.com/facebookresearch/drqv2)"""


class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST


class ActionRepeatWrapper(dm_env.Environment):
    def __init__(self, env, num_repeats):
        self._env = env
        self._num_repeats = num_repeats

    def step(self, action):
        reward = 0.0
        discount = 1.0
        for i in range(self._num_repeats):
            time_step = self._env.step(action)
            reward += (time_step.reward or 0.0) * discount
            discount *= time_step.discount
            if time_step.last():
                break

        return time_step._replace(reward=reward, discount=discount)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class FrameStackWrapper(dm_env.Environment):
    def __init__(self, env, num_frames, pixels_key='pixels'):
        self._env = env
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)
        self._pixels_key = pixels_key

        wrapped_obs_spec = env.observation_spec()
        assert pixels_key in wrapped_obs_spec

        pixels_shape = wrapped_obs_spec[pixels_key].shape
        if len(pixels_shape) == 4:
            pixels_shape = pixels_shape[1:]
        self._obs_spec = specs.BoundedArray(shape=np.concatenate(
            [[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0),
            dtype=np.uint8,
            minimum=0,
            maximum=255,
            name='observation')

    def _transform_observation(self, time_step):
        assert len(self._frames) == self._num_frames
        obs = np.concatenate(list(self._frames), axis=0)
        return time_step._replace(observation=obs)

    def _extract_pixels(self, time_step):
        pixels = time_step.observation[self._pixels_key]
        if len(pixels.shape) == 4:
            pixels = pixels[0]
        return pixels.transpose(2, 0, 1).copy()

    def reset(self):
        time_step = self._env.reset()
        pixels = self._extract_pixels(time_step)
        for _ in range(self._num_frames):
            self._frames.append(pixels)
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        pixels = self._extract_pixels(time_step)
        self._frames.append(pixels)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env
        wrapped_action_spec = env.action_spec()
        self._action_spec = specs.BoundedArray(wrapped_action_spec.shape,
                                               dtype,
                                               wrapped_action_spec.minimum,
                                               wrapped_action_spec.maximum,
                                               'action')

    def step(self, action):
        action = action.astype(self._env.action_spec().dtype)
        return self._env.step(action)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._action_spec

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

    def reset(self):
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(observation=time_step.observation,
                                step_type=time_step.step_type,
                                action=action,
                                reward=time_step.reward or 0.0,
                                discount=time_step.discount or 1.0)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class TimeStepToGymWrapper(object):
    def __init__(self, env, domain, task, action_repeat, modality):
        try:  # pixels
            obs_shp = env.observation_spec().shape
            assert modality == 'pixels'
        except:  # state
            obs_shp = []
            for v in env.observation_spec().values():
                try:
                    shp = np.prod(v.shape)
                except:
                    shp = 1
                obs_shp.append(shp)
            obs_shp = (np.sum(obs_shp, dtype=np.int32),)
            assert modality != 'pixels'
        act_shp = env.action_spec().shape
        obs_dtype = np.float32 if modality != 'pixels' else np.uint8
        self.observation_space = gym.spaces.Box(
            low=np.full(
                obs_shp,
                -np.inf if modality != 'pixels' else env.observation_spec().minimum,
                dtype=obs_dtype),
            high=np.full(
                obs_shp,
                np.inf if modality != 'pixels' else env.observation_spec().maximum,
                dtype=obs_dtype),
            shape=obs_shp,
            dtype=obs_dtype,
        )
        self.action_space = gym.spaces.Box(
            low=np.full(act_shp, env.action_spec().minimum),
            high=np.full(act_shp, env.action_spec().maximum),
            shape=act_shp,
            dtype=env.action_spec().dtype)
        self.env = env
        self.domain = domain
        self.task = task
        self.ep_len = 1000 // action_repeat
        self.modality = modality
        self.t = 0

    @property
    def unwrapped(self):
        return self.env

    @property
    def reward_range(self):
        return None

    @property
    def metadata(self):
        return None

    def _obs_to_array(self, obs):
        if self.modality != 'pixels':
            return np.concatenate([v.flatten() for v in obs.values()])
        return obs

    def reset(self):
        self.t = 0
        return self._obs_to_array(self.env.reset().observation)

    def step(self, action):
        self.t += 1
        time_step = self.env.step(action)
        info = {'is_success': self.env.is_success(), 'success': self.env.is_success()}
        return self._obs_to_array(
            time_step.observation), time_step.reward, time_step.last() or self.t == self.ep_len, info

    def render(self, mode='rgb_array', width=384, height=384, camera_id=0):
        camera_id = dict(quadruped=2).get(self.domain, camera_id)
        return self.env.physics.render(height, width, camera_id)


class DefaultDictWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, defaultdict(float, info)


class LeggedEnvWrapper(object):
    """Wrapper for quadruped locomotion."""
    def __init__(self, env):
        assert env.num_envs == 1
        input_space = env.get_input_space()
        output_space = env.get_output_space()
        self.obs_keys = ["proprio_obs", "privileged_infos"]
        obs_shape = (sum([input_space[k] for k in self.obs_keys]),)
        act_shape = (output_space,)
        self.observation_space = gym.spaces.Box(
            low=np.full(obs_shape, -np.inf, dtype=np.float32),
            high=np.full(obs_shape, np.inf, dtype=np.float32),
            shape=obs_shape,
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=np.full(act_shape, -1.),
            high=np.full(act_shape, 1.),
            shape=act_shape,
            dtype=np.float32,
        )
        self.env = env
        self.ep_len = 500
        self.t = 0

    @property
    def unwrapped(self):
        return self.env

    def reset(self):
        self.t = 0
        obs_dict = self.env.reset()
        return self._transform_obs(obs_dict)

    def step(self, action):
        self.t += 1
        a = torch.from_numpy(action) if isinstance(action, np.ndarray) else action
        obs, reward, done, info = self.env.step(a * self.env.clip_actions)
        done = info['fake_done']
        return self._transform_obs(obs), reward[0], done[0], info

    def render(self, mode='rgb_array', width=384, height=384, camera_id=0):
        return self.env.robot.log_render().cpu().numpy()

    def _transform_obs(self, obs_dict):
        return torch.concatenate([obs_dict[k][0] for k in self.obs_keys], dim=-1)


def make_xarm_env(cfg):
    """Make an xArm environmente."""
    import simxarm
    task = cfg.task[len("xarm_"):]
    obs_mode = 'rgb' if cfg.modality == 'pixels' else cfg.modality

    env = simxarm.make(task=task,
                       obs_mode=obs_mode,
                       image_size=cfg.get("img_size", 84),
                       action_repeat=cfg.get("action_repeat", 1),
                       frame_stack=cfg.get("frame_stack", 1),
                       seed=cfg.seed)
    # Convenience
    if obs_mode == 'all':
        cfg.obs_shape = {}
        for k in env.observation_space:
            cfg.obs_shape[k] = tuple(int(x) for x in env.observation_space[k].shape)
    else:
        cfg.obs_shape = tuple(int(x) for x in env.observation_space.shape)
    cfg.action_shape = tuple(int(x) for x in env.action_space.shape)
    cfg.action_dim = env.action_space.shape[0]

    for k in ['obs_shape', 'action_shape', 'action_dim']:
        print(k, getattr(cfg, k))

    return env


def make_d4rl_env(cfg):
    """Make an environment from D4RL."""
    import d4rl
    env = gym.make(cfg.task)
    env = gym.wrappers.RescaleAction(env, -1, 1)
    env = gym.wrappers.ClipAction(env)

    cfg.obs_shape = tuple(int(x) for x in env.observation_space.shape)
    cfg.action_shape = tuple(int(x) for x in env.action_space.shape)
    cfg.action_dim = env.action_space.shape[0]

    return env


def make_legged_env(cfg):
    """Make an environment for quadruped locomotion."""
    from isaacgym import gymapi
    from isaacgym import gymutil
    from complex_gym.robots import build_robot
    from complex_gym.surroundings import build_surrounding
    from complex_gym.tasks import build_task
    from complex_gym.envs.vec_env import VecEnv

    if "terrain" in cfg.task:
        env_cfg_path = "cfgs/legged_envs/go1_terrain.yaml"
    elif "fast" in cfg.task:
        env_cfg_path = "cfgs/legged_envs/go1_plane_fast.yaml"
    else:
        env_cfg_path = "cfgs/legged_envs/go1_plane.yaml"
    with open(env_cfg_path, "r") as f:
        env_cfg = yaml.load(f, Loader=yaml.SafeLoader)

    env_cfg["log_video"] = cfg["save_video"]

    def parse_sim_params(env_cfg):
        env_cfg["physics_engine"] = gymapi.SIM_PHYSX  # Default to PhysX

        sim_params = gymapi.SimParams()
        sim_params.dt = 1. / 400.
        sim_params.num_client_threads = env_cfg.get("slices", 0)

        sim_params.use_gpu_pipeline = True
        sim_params.physx.use_gpu = True

        # if sim options are provided in env_cfg, parse them and update/override above:
        if "sim" in env_cfg:
            gymutil.parse_sim_config(env_cfg["sim"], sim_params)

        return sim_params

    sim_params = parse_sim_params(env_cfg)

    rl_device = cfg["device"]
    env_cfg["env"]["num_envs"] = 1
    env_cfg["env"]["env_rows"] = 1
    env_cfg["env"]["env_cols"] = 1

    surrounding = build_surrounding(
        env_cfg["surrounding"]["name"],
        env_cfg,
        cfg["device"],
    )
    task = build_task(
        env_cfg["task"]["name"],
        env_cfg,
        cfg["device"],
    )
    robot = build_robot(
        env_cfg["robot"]["name"],
        surrounding,
        task,
        env_cfg,
        sim_params=sim_params,
    )

    env = VecEnv(
        robot=robot,
        task=task,
        surrounding=surrounding,
        cfg=env_cfg,
        robot_device=cfg["device"],
        rl_device=rl_device
    )

    env = LeggedEnvWrapper(env)

    cfg.obs_shape = tuple(int(x) for x in env.observation_space.shape)
    cfg.action_shape = tuple(int(x) for x in env.action_space.shape)
    cfg.action_dim = env.action_space.shape[0]

    return env


def make_env(cfg):
    """Make a task environment with cfg"""
    if cfg.task.startswith("xarm"):
        cfg.domain = "xarm"
        return make_xarm_env(cfg)
    elif cfg.task.startswith("legged"):
        cfg.domain = "legged"
        return make_legged_env(cfg)
    else:
        cfg.domain = "d4rl"
        return make_d4rl_env(cfg)
