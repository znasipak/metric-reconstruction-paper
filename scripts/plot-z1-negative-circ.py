import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Computer Modern"]
plt.rcParams["font.size"] = 18

if __name__ == "__main__":
    # Load the data from the file
    file = os.path.join(os.path.dirname(__file__), "..", "results", "z1-circular.csv")
    data_df = pd.read_csv(file)
    data = data_df[['p', 'z1']].to_numpy()
    
    # Define the psep value
    psep = 1.1817646130335848
    
    # Plot the data
    plt.plot(data[:, 0] - psep, data[:, 1], '.', label = 'z1')
    plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    plt.ylabel("$\\langle \\tilde{z}_1 \\rangle_t$")
    plt.xlabel("$p-p_\\mathrm{ISO}$")
    plt.savefig("figures/z1-negative-circ-corrected.pdf", bbox_inches='tight')