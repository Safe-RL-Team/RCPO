from datetime import datetime
import gym
import torch
from collections import OrderedDict

from safe_rl_stable_baselines3 import PPO, RCPPO
from hyperparameters import half_cheetah_params

### Create python env
#   0. For mujoco envs, you need to install mujoco (mujoco-py already defined in pyproject.toml)
#      https://github.com/openai/mujoco-py/
#   1. poetry shell
#   2. python setup.py install
#   3. poetry install
#   We'll need to modify the gym env
#   install gym-0.21.0 package
#   4. cd gym-0.21.0 && pip install -e .


def main():
    ### Example for cartpole env
    #   constraint is position of cartpole has to be to the left of the upper bound alpha
    # env = gym.make("CartPole-v1")

    # model = RCPPO(
    #     "MlpPolicy",
    #     env,
    #     verbose=1,
    #     constraint_alpha=-2,
    #     lr_constraint_lambda_decay_threshold=0.3,
    #     constant_constraint_lambda=None,
    #     lr_constraint_lambda=0.05,  # 0.02
    #     learning_rate=0.03,
    #     # tensorboard_log="/home/tuananhroman/tu/stable-baselines3/tensorboard",
    #     use_wandb=True,
    # )

    # model.learn(total_timesteps=100_000)

    ### Example for halfcheetah env
    env = gym.make("HalfCheetah-v3")
    save_model = True

    model = RCPPO(
        env=env,
        verbose=1,
        constraint_alpha=0.25,
        lr_constraint_lambda_decay_threshold=None,
        constant_constraint_lambda=1e-5,
        lr_constraint_lambda=5e-7,
        use_wandb=False,
        # tensorboard_log="/home/tuananhroman/tu/stable-baselines3/tensorboard",
        **half_cheetah_params,
    )

    model.learn(total_timesteps=100_000)

    if save_model:
        # get name of gym env
        env_name = env.spec.id
        # get current date and time
        now = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        # model name is the name of the env, the lambda value, and a time stamp
        model_name = f"{env_name}_{model.rollout_buffer.constraint_lambda}_{now}"

        model.save(path=f"runs/{model_name}")


if __name__ == "__main__":
    main()
