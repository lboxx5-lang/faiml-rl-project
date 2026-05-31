import argparse

import gymnasium as gym
import panda_gym  
from stable_baselines3 import PPO, SAC
from rand_wrapper import RandomizationWrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO/SAC on PandaPush-v3")
    parser.add_argument(
        "--algo",
        type=str,
        default="ppo",
        choices=["ppo", "sac"],
        help="RL algorithm",
    )
    parser.add_argument(
        "--sampling-strategy",
        type=str,
        default="none",
        choices=["none", "udr", "adr"],
        help="Sampling strategy for the object mass",
    )
    parser.add_argument(
        "--mass-min",
        type=float,
        default=0.5,
        help="Lower bound of randomization mass range",
    )
    parser.add_argument(
        "--mass-max",
        type=float,
        default=4.0,
        help="Upper bound of randomization mass range",
    )
    parser.add_argument(
        "--env-type",
        type=str,
        default="source",
        choices=["source", "target"],
        help="PandaPush environment type",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500_000,
        help="Number of training timesteps",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mass_min >= args.mass_max:
        raise ValueError("mass-min must be lower than mass-max")

    env = gym.make(
        "PandaPush-v3",
        type=args.env_type,
        reward_type="dense",
    )

    if args.sampling_strategy in ["udr", "adr"]:
        env = RandomizationWrapper(
            env,
            mass_range=(args.mass_min, args.mass_max),
            mode=args.sampling_strategy,
        )

    if args.algo == "ppo":
        model = PPO(
            policy="MultiInputPolicy",
            env=env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            ent_coef=0.01,
            seed=args.seed,
            tensorboard_log="./tensorboard_logs/"
        )
    else:
        model = SAC(
            policy="MultiInputPolicy",
            env=env,
            verbose=1,
            learning_rate=3e-4, 
            batch_size=256,      
            gamma=0.95,          
            tau=0.01,            
            ent_coef="auto",
            train_freq=1,
            gradient_steps=1,
            seed=args.seed,
            tensorboard_log="./tensorboard_logs/"
        )

    model.learn(total_timesteps=args.timesteps)

    save_name = (
        f"tuned_models/{args.algo}_push_{args.sampling_strategy}_{args.env_type}_"
        f"{args.timesteps // 1000}k_tuned_seed{args.seed}"
    )
    model.save(save_name)
    print(f"Saved model: {save_name}.zip")

    env.close()


if __name__ == "__main__":
    main()