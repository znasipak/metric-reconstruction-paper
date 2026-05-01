import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from matplotlib import rc
import matplotlib as mpl
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

df_u1_num_data = pd.read_csv("u1-numerical-comparison.csv")
df_u1_pn_data = pd.read_csv("u1-pn-comparison.csv")

u1_num_data = df_u1_num_data.to_numpy().reshape(4, 9, 4).swapaxes(0, 1)
u1_pn_data = df_u1_pn_data.to_numpy().reshape(9, 4, 9, 4).swapaxes(1, 2)

e_idx = 3
p_idx_start = 1
min_pn_order = 4
max_pn_order = 8

color_list = mpl.cm.viridis(np.linspace(0, 0.98, max_pn_order - min_pn_order + 1))

fig, ax = plt.subplots(1, 2, sharey = True, figsize = (10, 4))
fig.set_tight_layout('tight')

markers = ["o-", "P-", "v-", "D-", "s-", "X-", "^-"]

for i in range(min_pn_order, max_pn_order + 1):
    e_idx = 0
    data_comp = np.abs(u1_pn_data[i, p_idx_start:, e_idx, 3] - u1_num_data[p_idx_start:, e_idx, 2]) / np.abs(u1_num_data[p_idx_start:, e_idx, 2])
    err_comp = np.abs( u1_pn_data[i, p_idx_start:, e_idx, 3] / u1_num_data[p_idx_start:, e_idx, 2] ** 2 ) * u1_num_data[p_idx_start:, e_idx, 3]
    ax[0].errorbar(u1_num_data[p_idx_start:, e_idx, 0] - 2, data_comp, yerr=err_comp, label=f"{i}PN", fmt = markers[i-min_pn_order], capsize = 3, color = color_list[i-min_pn_order], markerfacecolor = "white")


    e_idx = 3
    data_comp = np.abs(u1_pn_data[i, p_idx_start:, e_idx, 3] - u1_num_data[p_idx_start:, e_idx, 2]) / np.abs(u1_num_data[p_idx_start:, e_idx, 2])
    err_comp = np.abs( u1_pn_data[i, p_idx_start:, e_idx, 3] / u1_num_data[p_idx_start:, e_idx, 2] ** 2 ) * u1_num_data[p_idx_start:, e_idx, 3]
    ax[1].errorbar(u1_num_data[p_idx_start:, e_idx, 0] - 2, data_comp, yerr=err_comp, label=f"{i}PN", fmt = markers[i-min_pn_order], capsize = 3, color = color_list[i-min_pn_order], markerfacecolor = "white")

ax[0].set_yscale("log")
ax[1].set_yscale("log")
ax[0].set_xscale("log")
ax[1].set_xscale("log")
ax[0].set_xlabel('$p - 2$')
ax[1].set_xlabel('$p - 2$')
ax[0].set_ylabel('relative difference $\\langle U_1 \\rangle$')
ax[0].text(0.85, 0.9, '$e = 0.1$', transform=ax[0].transAxes, color='black', bbox=dict(facecolor='none', edgecolor='black', lw = 0.5, boxstyle='round,pad=1'))
ax[1].text(0.85, 0.9, '$e = 0.6$', transform=ax[1].transAxes, color='black', bbox=dict(facecolor='none', edgecolor='black', lw = 0.5, boxstyle='round,pad=1'))
ax[1].legend(ncol = 2, loc = "lower left")

plt.savefig("../figures/pn-comparison-8PN.pdf", bbox_inches = "tight", dpi = 300)