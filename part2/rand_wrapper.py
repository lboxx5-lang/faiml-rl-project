
from collections import deque

import gymnasium as gym
import numpy as np

class RandomizationWrapper(gym.Wrapper):
    """
    Wrapper that applies randomization to the environment.
    """
    def __init__(
        self,
        env,
        mass_range=(1.0, 1.0),
        mode="none",
    ):
        super().__init__(env)

        self.mode = mode
        self.mass_range = mass_range

        # global limits
        self.mass_min_limit, self.mass_max_limit = mass_range

        # active range (for UDR this stays fixed, for ADR this is adapted)
        self.mass_min = float(self.mass_min_limit)
        self.mass_max = float(self.mass_max_limit)
        self.last_sample_type = "none"

        # ADR bookkeeping
        self.success_window = deque(maxlen=20)
        self.adr_step = 0.1

    # -----------------------
    # Mass Sampling
    # -----------------------

    def _sample_mass(self):

        if self.mode == "none":
            self.last_sample_type = "none"
            return None
        elif self.mode in ["udr", "adr"]:
            self.last_sample_type = "uniform"
            return float(np.random.uniform(self.mass_min, self.mass_max))
        else:
            raise NotImplementedError(f"Sampling strategy '{self.mode}' is not implemented yet.")

    def _update_adr_range(self):
        if len(self.success_window) < self.success_window.maxlen:
            return

        success_rate = float(np.mean(self.success_window))

        # Curriculum-like update: expand if easy, contract if too hard.
        if success_rate > 0.8:
            self.mass_min = max(self.mass_min_limit, self.mass_min - self.adr_step)
            self.mass_max = min(self.mass_max_limit, self.mass_max + self.adr_step)
        elif success_rate < 0.2:
            center = 0.5 * (self.mass_min + self.mass_max)
            half_width = max(0.1, 0.5 * (self.mass_max - self.mass_min) - self.adr_step)
            self.mass_min = max(self.mass_min_limit, center - half_width)
            self.mass_max = min(self.mass_max_limit, center + half_width)

    def step(self, action):

        obs, reward, terminated, truncated, info = self.env.step(action)

        done = terminated or truncated

        if done and self.mode == "adr":
            if isinstance(info, dict) and "is_success" in info:
                self.success_window.append(float(info["is_success"]))
                self._update_adr_range()

        return obs, reward, terminated, truncated, info

    # -----------------------
    # Reset
    # -----------------------

    def reset(self, **kwargs):

        new_mass = self._sample_mass()

        if new_mass is not None:

            sim = self.env.unwrapped.task.sim
            object_body_id = sim._bodies_idx["object"]

            sim.physics_client.changeDynamics(
                bodyUniqueId=object_body_id,
                linkIndex=-1,
                mass=float(new_mass),
            )

            print(
                f"[{self.mode}] mass={new_mass:.2f} "
                f"range=[{self.mass_min:.2f},{self.mass_max:.2f}] "
                f"type={self.last_sample_type}"
            )

        return super().reset(**kwargs)
