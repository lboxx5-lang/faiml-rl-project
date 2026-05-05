import gymnasium as gym
import mujoco

env = gym.make("Hopper-v4")

print("Observation space:")
print(env.observation_space)

print("\nAction space:")
print(env.action_space)

print("\nUnwrapped env:")
print(type(env.unwrapped))

model = env.unwrapped.model

print("\nBody names and masses:")
for i in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    mass = model.body_mass[i]
    print(i, name, mass)

print("\nNumber of DoFs:")
print(model.nv)

print("\nNumber of actuators:")
print(model.nu)

env.close()
