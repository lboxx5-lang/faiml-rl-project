"""Training loop for REINFORCE and Actor-Critic on Hopper-v4.

One on-policy update per episode. Hyperparameters are passed via CLI.

Examples:
    python3 train.py --algorithm reinforce --baseline 0.0  --n-episodes 10000
    python3 train.py --algorithm reinforce --baseline 10.0 --n-episodes 10000
    python3 train.py --algorithm ac                        --n-episodes 10000

Outputs (auto-named from hyperparameters):
    returns_<tag>.csv   per-episode return, length, elapsed time
    model_<tag>.pt      final policy weights
"""

import argparse
import csv
import os
import time

import gymnasium as gym
import numpy as np
import torch

from agent import Agent, Policy


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--algorithm",
        type=str,
        default="reinforce",
        choices=["reinforce", "ac"],
    )
    p.add_argument("--n-episodes", type=int, default=10000)
    p.add_argument("--print-every", type=int, default=100)
    p.add_argument(
        "--baseline",
        type=float,
        default=0.0,
    )
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--save",
        type=str,
        default=None,
    )
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def main():
    args = parse_args()

    # Seed everything for reproducibility.
    env = gym.make("Hopper-v4")
    env.reset(seed=args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    obs_dim = env.observation_space.shape[-1]
    act_dim = env.action_space.shape[-1]

    # Build network and agent.
    policy = Policy(obs_dim, act_dim)
    agent = Agent(
        policy, device=args.device, baseline=args.baseline, algorithm=args.algorithm
    )

    # Override Adam learning rate (Agent uses 1e-3 by default).
    for g in agent.optimizer.param_groups:
        g["lr"] = args.lr

    # File names derived from hyperparameters, so runs do not overwrite each other.
    tag = f"{args.algorithm}_b{args.baseline}_s{args.seed}"
    csv_path = f"returns_{tag}.csv"
    save_path = args.save if args.save is not None else f"model_{tag}.pt"

    print(
        f"Training {args.algorithm.upper()} on Hopper-v4 | "
        f"baseline={args.baseline} | lr={args.lr} | seed={args.seed}"
    )
    print(f"obs_dim={obs_dim}  act_dim={act_dim}")
    print(f"Outputs: {csv_path}  |  {save_path}")

    returns_history = []
    lengths_history = []
    times_history = []
    t0 = time.time()

    # ---- Main loop: one episode per iteration, one update per episode ----
    for ep in range(args.n_episodes):
        state, _ = env.reset()
        done = False
        ep_return = 0.0
        ep_len = 0

        # Roll out one full episode.
        while not done:
            action, log_prob = agent.get_action(state)
            action_np = action.detach().cpu().numpy()
            next_state, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated
            agent.store_outcome(state, next_state, log_prob, reward, done)
            state = next_state
            ep_return += reward
            ep_len += 1

        # One gradient step using the just-collected episode.
        agent.update_policy()

        returns_history.append(ep_return)
        lengths_history.append(ep_len)
        times_history.append(time.time() - t0)

        # Print rolling stats every K episodes.
        if (ep + 1) % args.print_every == 0:
            w = args.print_every
            avg_r = float(np.mean(returns_history[-w:]))
            std_r = float(np.std(returns_history[-w:]))
            avg_l = float(np.mean(lengths_history[-w:]))
            print(
                f"Ep {ep+1:6d} | ret={avg_r:7.2f} (std {std_r:6.2f}) | "
                f"len={avg_l:5.1f} | elapsed {times_history[-1]:6.1f}s"
            )

    # Save weights and per-episode CSV.
    torch.save(policy.state_dict(), save_path)
    print(f"Saved model to {os.path.abspath(save_path)}")

    with open(csv_path, "w", newline="") as f:
        out = csv.writer(f)
        out.writerow(["episode", "return", "length", "elapsed_s"])
        for i, (r, l, t) in enumerate(
            zip(returns_history, lengths_history, times_history)
        ):
            out.writerow([i, r, l, t])
    print(f"Saved returns to {os.path.abspath(csv_path)}")

    env.close()


if __name__ == "__main__":
    main()
