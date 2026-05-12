import argparse
import gymnasium as gym
import panda_gym
from stable_baselines3 import SAC
from rand_wrapper import RandomizationWrapper

def main() -> None:
    # Hardcoded arguments for the 500k Super UDR run
    algo = "sac"
    sampling_strategy = "udr"
    mass_min = 0.5
    mass_max = 4.0
    env_type = "source"
    timesteps = 500_000
    seed = 0

    env = gym.make("PandaPush-v3", type=env_type, reward_type="dense")
    env = RandomizationWrapper(env, mass_range=(mass_min, mass_max), mode=sampling_strategy)

    # Il TUNING: Rete maggiorata
    policy_kwargs = dict(net_arch=[512, 512])

    model = SAC(
        policy="MultiInputPolicy",
        env=env,
        verbose=1,
        learning_rate=1e-4,
        batch_size=1024,           # Batch grande per l'UDR
        gamma=0.99,
        tau=0.005,
        ent_coef="auto",
        train_freq=64,             # Accumula 64 step
        gradient_steps=64,         # Aggiorna 64 volte
        policy_kwargs=policy_kwargs,
        seed=seed,
        tensorboard_log="./tensorboard_logs/"
    )

    print("Inizio addestramento Super UDR a 500k step...")
    model.learn(total_timesteps=timesteps)

    save_name = f"{algo}_push_{sampling_strategy}_{env_type}_{timesteps // 1000}k_seed{seed}_TUNED"
    model.save(save_name)
    print(f"Modello salvato: {save_name}.zip")
    env.close()

if __name__ == "__main__":
    main()