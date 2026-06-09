"""Roll out a trained policy and save rendered episodes as a video.

Output format is chosen by file extension:
    .mp4   needs imageio-ffmpeg (pip install imageio-ffmpeg)
    .gif   works out of the box via Pillow

Examples:
    python3 eval_video.py --model model_ac_b0.0_s0.pt --out ac.mp4
    python3 eval_video.py --model model_reinforce_b0.0_s0.pt --out reinforce.gif
"""

import argparse

import gymnasium as gym
import imageio
import torch

from agent import Agent, Policy


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to a .pt checkpoint produced by train.py",
    )
    p.add_argument(
        "--out", type=str, default="hopper.mp4", help="Output video path (.mp4 or .gif)"
    )
    p.add_argument(
        "--episodes", type=int, default=3, help="Number of episodes to record"
    )
    p.add_argument(
        "--fps", type=int, default=30, help="Output framerate (one frame per env step)"
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample from the policy (default: take the Gaussian mean)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # rgb_array gives us numpy frames; works headless (no display needed).
    env = gym.make("Hopper-v4", render_mode="rgb_array")
    env.reset(seed=args.seed)

    # Build a Policy with the right shapes, then load saved weights.
    policy = Policy(env.observation_space.shape[-1], env.action_space.shape[-1])
    policy.load_state_dict(torch.load(args.model, map_location="cpu"))
    policy.eval()
    agent = Agent(policy)

    frames = []
    for ep in range(args.episodes):
        state, _ = env.reset()
        done = False
        total = 0.0
        steps = 0

        # Roll out one episode, capturing one frame per step.
        while not done:
            with torch.no_grad():
                # evaluation=True -> Gaussian mean (deterministic)
                # evaluation=False -> sample (stochastic, like training)
                action, _ = agent.get_action(state, evaluation=not args.stochastic)
            state, r, term, trunc, _ = env.step(action.cpu().numpy())
            frames.append(env.render())
            total += r
            steps += 1
            done = term or trunc

        print(f"ep {ep+1}: return = {total:.2f} ({steps} steps)")

    env.close()
    imageio.mimsave(args.out, frames, fps=args.fps)
    print(f"Saved {len(frames)} frames -> {args.out}")


if __name__ == "__main__":
    main()
