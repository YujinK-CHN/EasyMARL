import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
    
def load_algorithm_data(algo_path, column):
    all_runs = []

    for file in os.listdir(algo_path):
        if file.endswith(".csv"):
            file_path = os.path.join(algo_path, file)
            df = pd.read_csv(file_path)

            if column not in df.columns:
                continue

            all_runs.append(df[column].values)

    if not all_runs:
        return None

    # make all runs same length
    min_len = min(len(run) for run in all_runs)
    all_runs = np.array([run[:min_len] for run in all_runs])

    return all_runs


def plot_results(base_dir, column, mode=None, shade=None):
    plt.figure()

    for algo in os.listdir(base_dir):
        algo_path = os.path.join(base_dir, algo)

        if not os.path.isdir(algo_path):
            continue

        all_runs = load_algorithm_data(algo_path, column)

        if all_runs is None:
            print(f"[WARNING] {algo} missing column: {column}")
            continue

        # ===== APPLY LOG MODE =====
        if mode == "log":
            eps = 1e-12
            all_runs = np.log10(all_runs + eps)

        mean_curve = np.mean(all_runs, axis=0)
        x = np.arange(len(mean_curve))

        plt.plot(x, mean_curve, label=algo)

        # ===== SHADING OPTIONS =====
        if shade == "variance":
            std_curve = np.std(all_runs, axis=0)
            lower = mean_curve - std_curve
            upper = mean_curve + std_curve

            plt.fill_between(x, lower, upper, alpha=0.2)

        elif shade == "min_max_generation":
            min_runs = []
            max_runs = []

            min_col = f"min_{column}"
            max_col = f"max_{column}"

            for file in os.listdir(algo_path):
                if file.endswith(".csv"):
                    df = pd.read_csv(os.path.join(algo_path, file))

                    if min_col in df.columns and max_col in df.columns:
                        min_runs.append(df[min_col].values)
                        max_runs.append(df[max_col].values)

            if min_runs and max_runs:
                min_len = min(len(r) for r in min_runs)
                min_runs = np.array([r[:min_len] for r in min_runs])
                max_runs = np.array([r[:min_len] for r in max_runs])

                # ===== APPLY LOG MODE HERE TOO =====
                if mode == "log":
                    eps = 1e-12
                    min_runs = np.log10(min_runs + eps)
                    max_runs = np.log10(max_runs + eps)

                avg_min = np.mean(min_runs, axis=0)
                avg_max = np.mean(max_runs, axis=0)

                x_shade = np.arange(len(avg_min))

                plt.fill_between(x_shade, avg_min, avg_max, alpha=0.2)

    plt.xlabel("Generation")

    if mode == "log":
        plt.ylabel("log10(Cost)")
    else:
        plt.ylabel("Cost")

    # ===== FIX SCALE USAGE =====
    if mode == "log":
        plt.yscale("linear")   

    plt.title(column)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    base_dir = "results/PPO"  # root folder
    

    column = "epsisode_reward"  # column to plot
    plot_results(base_dir, column, shade="variance")