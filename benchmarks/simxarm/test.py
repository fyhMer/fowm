import numpy as np
import simxarm


def test():
    """
    Unit tests for the simxarm package.
    """
    print('Testing simxarm package...')
    print('Tasks:')
    print(simxarm.TASKS)

    # Test each environment
    for task in simxarm.TASKS:
        env = simxarm.make(task)
        print('\nInitialized environment: {}'.format(task))
        print('Observation space: {}'.format(env.observation_space.shape))
        print('Action space: {}'.format(env.action_space.shape))

        # Test reset
        obs = env.reset()

        # Test step
        action = env.action_space.sample()
        next_obs, reward, done, info = env.step(action)
        assert obs.shape == next_obs.shape
        assert obs.shape == env.observation_space.shape
        assert len(action) == len(simxarm.TASKS[task]['action_space'])
        assert 'is_success' in info.keys()

        # Test render
        frame = env.render(mode='rgb_array', width=384, height=384)
        assert frame.shape == (384, 384, 3)

        # Test close
        env.close()

    # Test input types
    print('\nInput types:')
    for obs_mode in ['state', 'rgb', 'all']:
        env = simxarm.make('lift', obs_mode=obs_mode)
        if obs_mode == 'all':
            print('Observation space {}: {}'.format(obs_mode, [space.shape for space in env.observation_space.spaces.values()]))
        else:
            print('Observation space {}: {}'.format(obs_mode, env.observation_space.shape))
        obs = env.reset()
        assert isinstance(obs, dict) or obs.shape == env.observation_space.shape
        env.close()

    # Test action repeat
    print('\nAction repeat:')
    for action_repeat in [1, 2]:
        env = simxarm.make('lift', action_repeat=action_repeat)
        env.reset()
        for _ in range(simxarm.TASKS['lift']['episode_length']//action_repeat):
            action = env.action_space.sample()
            _, _, done, _ = env.step(action)
        assert done, 'Environment should be done after {} steps'.format(simxarm.TASKS['lift']['episode_length']//action_repeat)
        print(f'Action repeat {action_repeat} returned done after {simxarm.TASKS["lift"]["episode_length"]//action_repeat} steps')
        env.close()


if __name__ == '__main__':
    test()
