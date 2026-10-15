
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from aircraft_env import AircraftReallocationEnv


BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

env = AircraftReallocationEnv()

# Check that the environment follows the Gymnasium API.
check_env(env, warn=True)

print("Starting PPO training...")

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=0.0003,
    n_steps=256,
    batch_size=64,
    gamma=0.99,
    verbose=1,
    seed=42,
)

model.learn(total_timesteps=20_000)

model_path = MODEL_DIR / "ppo_aircraft"
model.save(str(model_path))

print("\nPPO TRAINING COMPLETE")
print("Saved model:", str(model_path) + ".zip")

env.close()
