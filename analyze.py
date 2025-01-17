import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def load_results(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)[1]['trials']

def analyze_results(results):
    summary = []
    for i, result in enumerate(results):
        alpha = result["alpha"]
        gamma = result["gamma"]
        epsilon_start = result["epsilon_start"]
        decay_step = result["decay_step"]
        avg_reward = result["average_reward"]
        std_dev_reward = result["std_dev_reward"]
        convergence_epoch = result.get("convergence_epoch", "N/A")        
        stability_diffs = result.get("stability_diffs", [])
        final_reward = result.get("rewards")[-1]

        summary.append({
            "Configuração": f"Setup {i+1}",
            "Alpha": alpha,
            "Gamma": gamma,
            "Epsilon Start": epsilon_start,
            "Decay Step": decay_step,
            "Recompensa Média": avg_reward,
            "Desvio Padrão": std_dev_reward,
            "Época de Convergência": convergence_epoch,
            "Estabilidade Final": stability_diffs[-1] if stability_diffs else "N/A",
            "Recompensa Final":final_reward
        })

    return pd.DataFrame(summary)
    
def plot_stability_diffs(results,threshold=1e-3):
    
    for i, result in enumerate(results):
        stability_diff = result["stability_diffs"]
        plt.figure(figsize=(10, 6))
        plt.plot(stability_diff, label="Variação na Política")
        plt.axhline(y=threshold, color='r', linestyle='--', label="Threshold")
        plt.title(f"Variação na Política por Época - Setup {i+1}")
        plt.xlabel("Época")
        plt.ylabel("Variação Média na Política")
        plt.grid()
        plt.legend()
        plt.show()   
    
def plot_convergence(rewards, window_size=1000):
    plt.figure(figsize=(12, 6))
    for i, reward in enumerate(rewards):
        smoothed_rewards = np.convolve(reward, np.ones(window_size)/window_size, mode='valid')
        plt.plot(smoothed_rewards, label=f'Setup {i+1}')

    plt.title('Convergência das Recompensas ao Longo das Iterações')
    plt.xlabel('Iterações')
    plt.ylabel('Recompensa Média Suavizada')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_policy_heatmap(policies, labels):
    for i, policy in enumerate(policies):
        plt.figure(figsize=(8, 4))
        sns.heatmap(policy, annot=True, cmap="coolwarm", cbar=True)
        plt.title(f"Policy Heatmap - Setup {i+1}: Alpha {labels[i]['alpha']}, Gamma {labels[i]['gamma']}")
        plt.show()

def plot_rewar_evalution(data):
    # Extract rewards for the setups
    setups_rewards = [trial["rewards"] for trial in data[:5]]

    # Calculate the average rewards every 10,000 interactions for each setup
    chunk_size = 10000
    setups_means = []
    for rewards in setups_rewards:
        chunks = [rewards[i:i + chunk_size] for i in range(0, len(rewards), chunk_size)]
        setups_means.append([np.mean(chunk) for chunk in chunks])

    # Plot the average rewards for each setup
    plt.figure(figsize=(12, 6))
    for i, setup_mean in enumerate(setups_means):
        plt.plot(setup_mean, label=f'Setup {i + 1}')

    plt.title('Average Rewards per 10,000 Interactions for Each Setup')
    plt.xlabel('Block of 10,000 Interactions')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_comparison(summary_df):
    """Compara as configurações usando um gráfico de barras."""
    # Comparar épocas de convergência
    converged = summary_df[summary_df["Época de Convergência"] != "Não Convergiu"]
    if not converged.empty:
        plt.figure(figsize=(10, 6))
        plt.bar(converged["Configuração"], converged["Época de Convergência"], color='blue', alpha=0.7)
        plt.title("Época de Convergência por Configuração")
        plt.xlabel("Configuração")
        plt.ylabel("Época de Convergência")
        plt.grid(axis="y")
        plt.show()

    # Comparar recompensas médias
    plt.figure(figsize=(10, 6))
    plt.bar(summary_df["Configuração"], summary_df["Recompensa Média"], color='green', alpha=0.7)
    plt.title("Recompensa Média por Configuração")
    plt.xlabel("Configuração")
    plt.ylabel("Recompensa Média")
    plt.grid(axis="y")
    plt.show()

def analyze():
    file_path = "results.json"  # Specify the correct path to your results file
    results = load_results(file_path)
    summary_df = analyze_results(results)

    rewards = [result['rewards'] for result in results]
    
    plot_rewar_evalution(results)
    
    plot_convergence(rewards)

    plot_stability_diffs(results)
    
    plot_comparison(summary_df)
    
    
    # Save summary to CSV
    summary_df.to_csv("summary_analysis.csv", index=False)
    print("Analysis summary saved to 'summary_analysis.csv'.")

if __name__ == "__main__":
    analyze()
