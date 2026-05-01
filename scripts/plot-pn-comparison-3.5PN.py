import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from matplotlib import rc
import matplotlib as mpl
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
rc('font', size=12)  # sets default font size
rc('axes', labelsize=12)  # axis labels
# rc('xtick', labelsize=11)  # x-axis tick labels
# rc('ytick', labelsize=11)  # y-axis tick labels
rc('legend', fontsize=10)  # legend

root_dir = os.path.dirname(os.path.abspath(__file__))
savepath = os.path.join(root_dir, "..", "figures", "pn-comparison-3.5PN.pdf")

numerical_comparison_filepath = os.path.join(root_dir, "..", "results", "z1-equatorial_pn_comparison.csv")
z1_pn_comparison_filepath = os.path.join(root_dir, "..", "results", "z1-pn-reference.csv")

df_num_data = pd.read_csv(numerical_comparison_filepath)
df_z1_pn_data = pd.read_csv(z1_pn_comparison_filepath)

df_num_a99_data = df_num_data[df_num_data["a"] == 0.99]

z1_num_data = df_num_a99_data[["p", "e", "z1", "z1_err"]].to_numpy().reshape(4, 9, 4).swapaxes(0, 1)
z1_pn_data = df_z1_pn_data[["p", "e", "z1"]].to_numpy().reshape(4, 9, 3).swapaxes(0, 1)

fig, ax = plt.subplots(1, 1, sharey = True, figsize = (4, 3))
fig.set_tight_layout('tight')
e_vals = z1_num_data[0, :, 1]

color_list = mpl.cm.viridis(np.linspace(0, 0.98, len(e_vals)))

markers = ["o-", "P-", "v-", "D-"]

p_idx_start = 1

for e_idx in range(4):
    data_comp = np.abs(z1_pn_data[p_idx_start:, e_idx, 2] - z1_num_data[p_idx_start:, e_idx, 2]) / np.abs(z1_num_data[p_idx_start:, e_idx, 2])
    err_comp = np.abs( z1_pn_data[p_idx_start:, e_idx, 2] / z1_num_data[p_idx_start:, e_idx, 2] ** 2 ) * z1_num_data[p_idx_start:, e_idx, 3]
    ax.errorbar(z1_num_data[p_idx_start:, e_idx, 0] - 2, data_comp, yerr=err_comp, label=f"$e = {{{e_vals[e_idx]:.1f}}}$", fmt = markers[e_idx], capsize = 3, color = color_list[e_idx], markerfacecolor = "white")

ax.set_yscale("log")
ax.set_xscale("log")
ax.set_xlabel('$p - 2$')
ax.set_ylabel('relative difference $\\langle \\tilde{z}_1 \\rangle_t$')
plt.legend(ncol = 1, loc = "lower left")

plt.savefig(savepath, bbox_inches = "tight", dpi = 300)