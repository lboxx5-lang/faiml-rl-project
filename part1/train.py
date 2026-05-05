"""Sample script for training a control policy on the Hopper environment.

Training loop for:
- REINFORCE without baseline
- REINFORCE with constant baseline

Actor-Critic can be added later using the same structure.
"""

import argparse
import time

import gymnasium as gym
import numpy as np
import torch

from agent import Policy, Agent


def evaluate_policy(env, agent, n_episodes=10):
    returns = []

    for _ in range(n_episodes):
        state, info = env.reset()
        done = False
        episode_return = 0.0

        while not done:
            action, _ = agent.get_action(state, evaluation=True)

            action_np = action.detach().cpu().numpy()
            action_np = np.clip(
                action_np,
                env.action_space.low,
                env.action_space.high,
            )

            state, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated

            episode_return += reward

        returns.append(episode_return)

    return float(np.mean(returns)), float(np.std(returns))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--env", type=str, default="Hopper-v4")
    parser.add_argument(
        "--algorithm",
        type=str,
        default="reinforce",
        choices=["reinforce", "reinforce_baseline"],
    )
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = gym.make(args.env)

    print("State space:", env.observation_space)
    print("Action space:", env.action_space)

    state_space = env.observation_space.shape[0]
    action_space = env.action_space.shape[0]

    policy = Policy(state_space, action_space)

    if args.algorithm == "reinforce":
        agent = Agent(policy, use_baseline=False)
    elif args.algorithm == "reinforce_baseline":
        agent = Agent(policy, use_baseline=True)
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm}")

    episode_returns = []
    episode_lengths = []

    start_time = time.time()

    for episode in range(args.episodes):
        state, info = env.reset(seed=args.seed + episode)

        done = False
        episode_return = 0.0
        episode_length = 0

        while not done:
            action, action_log_prob = agent.get_action(state, evaluation=False)

            action_np = action.detach().cpu().numpy()
            action_np = np.clip(
                action_np,
                env.action_space.low,
                env.action_space.high,
            )

            next_state, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated

            agent.store_outcome(
                state=state,
                next_state=next_state,
                action_log_prob=action_log_prob,
                reward=reward,
                done=done,
            )

            state = next_state
            episode_return += reward
            episode_length += 1

        loss = agent.update_policy()

        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)

        if (episode + 1) % args.log_interval == 0:
            avg_return = np.mean(episode_returns[-args.log_interval :])
            avg_length = np.mean(episode_lengths[-args.log_interval :])
            elapsed = time.time() - start_time

            print(
                f"Episode {episode + 1:5d} | "
                f"Return: {episode_return:9.2f} | "
                f"Avg return: {avg_return:9.2f} | "
                f"Avg length: {avg_length:7.1f} | "
                f"Loss: {loss:10.2f} | "
                f"Elapsed: {elapsed:7.1f}s"
            )

    eval_mean, eval_std = evaluate_policy(env, agent, n_episodes=args.eval_episodes)

    print("\nFinal evaluation")
    print(f"Algorithm: {args.algorithm}")
    print(f"Evaluation episodes: {args.eval_episodes}")
    print(f"Mean return: {eval_mean:.2f}")
    print(f"Std return: {eval_std:.2f}")

    env.close()

    returns_file = f"returns_{args.algorithm}.npy"
    model_file = f"policy_{args.algorithm}.pt"

    np.save(returns_file, np.array(episode_returns))
    torch.save(policy.state_dict(), model_file)

    print(f"\nSaved {returns_file}")
    print(f"Saved {model_file}")


if __name__ == "__main__":
    main()
