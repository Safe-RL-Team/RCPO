import gym

from stable_baselines3 import PPO, RCPPO

### Create python env
#   0. For mujoco envs, you need to install mujoco (mujoco-py already defined in pyproject.toml)
#      https://github.com/openai/mujoco-py/
#   1. poetry shell
#   2. python setup.py install
#   3. poetry install
#   We'll need to modify the gym env
#   install gym-0.21.0 package
#   4. cd gym-0.21.0 && pip install -e .


### Example for cartpole env
#   constraint is position of cartpole has to be to the left of the upper bound alpha
env = gym.make("CartPole-v1")
model = RCPPO(
    "MlpPolicy", env, verbose=1, constraint_alpha=-2, lr_constraint_lambda=0.1
)

### Example for halfcheetah env
env = gym.make("HalfCheetah-v2")
# model = RCPPO("MlpPolicy", env, verbose=1, constraint_alpha=1, lr_constraint_lambda=0.1)

model.learn(total_timesteps=100_000)

obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()

env.close()
