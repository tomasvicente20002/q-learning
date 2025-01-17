import numpy as np
import matplotlib.pyplot as plt
import json
from gridworld import GridWorld
from qlearning import *
from datetime import datetime
from analyze import analyze

TOP_EPOCH = 300000  # Número total de iterações
PRINT_EPOCH = 5000  # Intervalo para monitorizar resultados


def serialize_results(state_matrix,reward_matrix, policy_matrix, results):
    serialized_results = []
    serialized_json = []
    serialized_json.append({
        "initial_env_config":{
        "state_matrix": state_matrix.tolist(),
        "reward_matrix":reward_matrix.tolist(),
        "initial_policy_matrix":policy_matrix.tolist(),
        }
    })
    
    for result in results:
        serialized_result = {
            "alpha": float(result["alpha"]),  # Converter para float
            "gamma": float(result["gamma"]),  # Converter para float
            "epsilon_start": float(result["epsilon_start"]),  # Converter para float
            "decay_step": int(result["decay_step"]),  # Converter para int
            "rewards": [float(r) for r in result["rewards"]],  # Converter lista de recompensas
            "average_reward": float(np.mean(result["rewards"])),  # Converter para float
            "std_dev_reward": float(np.std(result["rewards"])),  # Converter para float
            "final_policy": result["final_policy"].tolist() if isinstance(result["final_policy"], np.ndarray) else result["final_policy"],
            "visit_counter_matrix": result["visit_counter_matrix"].tolist() if isinstance(result["visit_counter_matrix"], np.ndarray) else result["visit_counter_matrix"],
            "steps_per_epoch": [int(steps) for steps in result["steps_per_epoch"]],  # Garantir que são inteiros
            "convergence_epoch": int(result["convergence_epoch"]) if result["convergence_epoch"] is not None else None,  # Garantir que é inteiro ou None
            "stability_diffs": [float(diff) for diff in result["stability_diffs"]],  # Converter lista de diferenças para float
        }
        serialized_results.append(serialized_result)
        
    serialized_json.append({"trials":serialized_results})
    return serialized_json


def train_q_learning(env, alpha, gamma, epsilon_start, decay_step, policy_matrix, state_action_matrix, visit_counter_matrix, threshold=1e-3):
    rewards_per_epoch = []
    previous_policy_matrix = np.copy(policy_matrix)
    convergence_epoch = None
    steps_per_epoch = []
    stability_diffs = []  # Guardar variações médias da política
    stability_counter = 0
    required_stable_epochs = 10  # Número de épocas consecutivas para convergência

    for epoch in range(TOP_EPOCH):
        observation = env.reset(exploring_starts=True)
        epsilon = return_decayed_value(epsilon_start, epoch, decay_step)
        total_reward = 0
        steps = 0

        for _ in range(1000):
            action = return_epsilon_greedy_action(policy_matrix, observation, epsilon)
            new_observation, reward, done = env.step(action)
            total_reward += reward
            state_action_matrix = update_state_action(state_action_matrix, visit_counter_matrix, observation, new_observation, action, reward, alpha, gamma)
            policy_matrix = update_policy(policy_matrix, state_action_matrix, observation)
            visit_counter_matrix = update_visit_counter(visit_counter_matrix, observation, action)
            observation = new_observation
            steps += 1
            if done:
                break

        rewards_per_epoch.append(total_reward)
        steps_per_epoch.append(steps)

        # Calcular a variação média da política
        mean_diff = np.nanmean(np.abs(policy_matrix - previous_policy_matrix))
        stability_diffs.append(mean_diff)

        # Verificar convergência com base na estabilidade da política
        if mean_diff < threshold:
            stability_counter += 1
            if convergence_epoch is None and stability_counter >= required_stable_epochs:
                convergence_epoch = epoch
        else:
            stability_counter = 0

        previous_policy_matrix = np.copy(policy_matrix)

        if epoch % PRINT_EPOCH == 0:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_message = (f"Data/Hora: {current_time}, Época: {epoch}, Epsilon: {epsilon:.6f}, "
                        f"Recompensa Média: {np.mean(rewards_per_epoch[-PRINT_EPOCH:]):.4f}, "
                        f"Variação Média na Política: {mean_diff:.6f}")


            with open("training_log.txt", "a") as log_file:
                log_file.write(log_message + "\n")

    return state_action_matrix, policy_matrix, rewards_per_epoch, steps_per_epoch, convergence_epoch, stability_diffs

def main():
    env = GridWorld(3, 4)
    state_matrix = np.zeros((3, 4))
    state_matrix[0, 3] = 1
    state_matrix[1, 3] = 1
    state_matrix[1, 1] = -1

    reward_matrix = np.full((3,4), -0.04)
    reward_matrix[0, 3] = 1  # Objetivo positivo
    reward_matrix[1, 3] = -1  # Estado terminal negativo
    transition_matrix = np.array([[0.8, 0.1, 0.0, 0.1], 
                                   [0.1, 0.8, 0.1, 0.0], 
                                   [0.0, 0.1, 0.8, 0.1], 
                                   [0.1, 0.0, 0.1, 0.8]])

    policy_matrix = np.random.randint(low=0, high=4, size=(3, 4)).astype(np.float32)
    policy_matrix[1, 1] = np.nan
    policy_matrix[0, 3] = policy_matrix[1, 3] = -1
    env.setStateMatrix(state_matrix)
    env.setRewardMatrix(reward_matrix)
    env.setTransitionMatrix(transition_matrix)

    state_action_matrix = np.zeros((4, 12))
    visit_counter_matrix = np.zeros((4, 12))
    setups = [
        {"alpha": 0.01, "gamma": 0.9, "epsilon_start": 0.2, "decay_step": 50000},
        {"alpha": 0.005, "gamma": 0.85, "epsilon_start": 0.2, "decay_step": 50000},        
        {"alpha": 0.1, "gamma": 0.95, "epsilon_start": 0.3, "decay_step": 100000},
        {"alpha": 0.001, "gamma": 0.5, "epsilon_start": 0.2, "decay_step": 50000},
        {"alpha": 0.005, "gamma": 0.8, "epsilon_start": 0.1, "decay_step": 25000}
    ]
    results = []

    for setup in setups:
        alpha = setup["alpha"]
        gamma = setup["gamma"]
        epsilon_start = setup["epsilon_start"]
        decay_step = setup["decay_step"]

        state_action_matrix, policy_matrix, rewards, steps_per_epoch, convergence_epoch, stability_diffs = train_q_learning(
            env, alpha, gamma, epsilon_start, decay_step, policy_matrix, state_action_matrix, visit_counter_matrix)

        results.append({
            "alpha": alpha,
            "gamma": gamma,
            "epsilon_start": epsilon_start,
            "decay_step": decay_step,
            "rewards": rewards,
            "final_policy": policy_matrix.tolist(),
            "visit_counter_matrix": visit_counter_matrix.tolist(),
            "steps_per_epoch": steps_per_epoch,
            "convergence_epoch": convergence_epoch,
            "stability_diffs": stability_diffs,
        })

    with open('results.json', 'w') as f:
        json.dump(serialize_results(state_matrix,reward_matrix, policy_matrix, results), f)

    analyze()

if __name__ == "__main__":
    main()
