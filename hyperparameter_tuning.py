import gym
import energym
import optuna
import numpy as np
import random
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from custom_policy import CustomDQNPolicy

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def objective(trial):
    """Optuna objective function for hyperparameter optimization"""
    
    # Hyperparameters to tune
    gamma = trial.suggest_float("gamma", 0.85, 0.99)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    buffer_size = trial.suggest_categorical("buffer_size", [100000, 500000, 1000000])
    learning_starts = trial.suggest_categorical("learning_starts", [10000, 20000, 35040])
    target_update_interval = trial.suggest_categorical("target_update_interval", [1000, 5000, 10000])
    exploration_fraction = trial.suggest_float("exploration_fraction", 0.1, 0.9)
    exploration_final_eps = trial.suggest_float("exploration_final_eps", 0.01, 0.1)
    
    # Create environment
    env = gym.make('Eplus-discrete-hot-v1')
    env.seed(SEED)
    env.action_space.seed(SEED)
    env.observation_space.seed(SEED)
    env = Monitor(env)
    
    # Create model with trial hyperparameters
    model = DQN(
        CustomDQNPolicy,  # Use our custom policy
        env,
        gamma=gamma,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        target_update_interval=target_update_interval,
        exploration_fraction=exploration_fraction,
        exploration_initial_eps=1.0,
        exploration_final_eps=exploration_final_eps,
        seed=SEED,
        verbose=0
    )
    
    # Train for a short period (adjust based on your computational resources)
    model.learn(total_timesteps=100000)
    
    # Evaluate
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=2)
    
    env.close()
    
    return mean_reward

def run_optimization(n_trials=20):
    """Run the hyperparameter optimization"""
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    
    print("Best hyperparameters:", study.best_params)
    print("Best reward:", study.best_value)
    
    return study.best_params

if __name__ == "__main__":
    best_params = run_optimization(n_trials=20)
    print(f"Best parameters: {best_params}")