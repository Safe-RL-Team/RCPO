import torch
import gym

from safe_rl_stable_baselines3 import RCPPO


def main():
    print("evaluating model")

    # Load pytorch model
    model_name = "HalfCheetah-v3_100_16-01-2023_16-21-26"
    model = RCPPO.load(f"runs/{model_name}")

    # Get env of model
    env = gym.make(model_name.split("_")[0])

    obs = env.reset()
    for _ in range(5000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()

    env.close()


if __name__ == "__main__":
    main()
