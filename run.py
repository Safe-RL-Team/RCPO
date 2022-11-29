# We'll need to modify the gym env
# install gym-0.21.0 package
# cd gym-0.21.0 && pip install -e .

import gym

from stable_baselines3 import PPO

### Create python env
#   0. For mujoco envs, you need to install mujoco (mujoco-py already defined in pyproject.toml)
#      https://github.com/openai/mujoco-py/
#   1. poetry shell
#   2. python setup.py install
#   3. poetry install

# env = gym.make("CartPole-v1")
env = gym.make("HalfCheetah-v2")

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)

obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()

env.close()
