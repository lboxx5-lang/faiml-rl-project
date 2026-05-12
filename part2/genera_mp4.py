import gymnasium as gym
import panda_gym
from stable_baselines3 import SAC
from gymnasium.wrappers import RecordVideo
import os

# Cartella dove verranno salvati i video
video_folder = "./video_report"

def record_model(model_path, env_type, prefix):
    # Creiamo l'ambiente in modalità 'rgb_array' (necessaria per il video)
    env = gym.make("PandaPush-v3", render_mode="rgb_array", type=env_type)
    
    # Applichiamo il wrapper per registrare
    env = RecordVideo(env, video_folder, name_prefix=prefix, 
                      episode_trigger=lambda x: True) # registra ogni episodio

    print(f"Registrazione modello: {model_path}...")
    model = SAC.load(model_path)

    obs, info = env.reset()
    for _ in range(1): # Registriamo 1 episodio per modello per brevità
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                done = True
    
    env.close()
    print(f"Video salvato in {video_folder}")

if __name__ == "__main__":
    # 1. Registra il fallimento (Reality Gap)
    # Usiamo il modello base testato sul Target (cubo da 5kg)
    try:
        record_model("sac_push_none_source_2000k_seed0.zip", "target", "01_FALLIMENTO_REALITY_GAP")
    except Exception as e:
        print(f"Errore registrazione fallimento: {e}")

    # 2. Registra il successo (UDR)
    # Usiamo il modello robusto testato sullo stesso Target
    try:
        record_model("sac_push_udr_source_2000k_seed0.zip", "target", "02_SUCCESSO_UDR")
    except Exception as e:
        print(f"Errore registrazione successo: {e}")