import os
import numpy as np
import matplotlib.pyplot as plt


def moving_average(x, window=20):
    """
    Compute moving average for smoother visualization.
    """
    if len(x) < window:
        return x

    return np.convolve(x, np.ones(window) / window, mode="valid")


def load_returns(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"Missing file: {filename}. " f"Run the corresponding training first."
        )

    return np.load(filename)


def main():
    reinforce_file = "returns_reinforce.npy"
    reinforce_baseline_file = "returns_reinforce_baseline.npy"

    reinforce_returns = load_returns(reinforce_file)
    reinforce_baseline_returns = load_returns(reinforce_baseline_file)

    window = 20

    reinforce_smooth = moving_average(reinforce_returns, window)
    reinforce_baseline_smooth = moving_average(reinforce_baseline_returns, window)

    plt.figure(figsize=(10, 6))

    plt.plot(reinforce_smooth, label="REINFORCE without baseline")

    plt.plot(reinforce_baseline_smooth, label="REINFORCE with constant baseline")

    plt.xlabel("Episode")
    plt.ylabel(f"Return moving average, window={window}")
    plt.title("Hopper training comparison")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("hopper_reinforce_comparison.png", dpi=300)
    plt.show()

    print("Saved hopper_reinforce_comparison.png")

    print("\nFinal statistics")
    print("-----------------------------------")
    print(
        "REINFORCE without baseline:",
        f"mean last 100 = {np.mean(reinforce_returns[-100:]):.2f}",
        f"std last 100 = {np.std(reinforce_returns[-100:]):.2f}",
    )
    print(
        "REINFORCE with baseline:",
        f"mean last 100 = {np.mean(reinforce_baseline_returns[-100:]):.2f}",
        f"std last 100 = {np.std(reinforce_baseline_returns[-100:]):.2f}",
    )


if __name__ == "__main__":
    main()
