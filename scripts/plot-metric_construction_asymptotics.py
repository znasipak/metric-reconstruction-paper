from pybhpt.geo import KerrGeodesic
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import matplotlib as mpl

from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

if __name__ == "__main__":
    # Load metadata file which tells us which data files to load
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "metric")
    df = pd.read_csv(os.path.join(data_dir, 'metric_metadata.csv'))

    # Select a specific system to plot
    a0 = 0.9
    ptemp = 8.0
    lmin = 2
    lmax = 30

    # Filter the DataFrame for the specific parameters
    df_subset = df.loc[(df['a'] == a0) & (df['p'] == ptemp) & (df['lmax'] == lmax)]
    gauges = df_subset['gauge'].unique()

    # Load the radial grid from the first file in the subset
    r_grid = np.load(os.path.join(data_dir, df_subset['rgrid_file'].values[0]))
    p0_loc = np.where(r_grid == ptemp)[0][0]
    lmax_temp = df_subset['lmax'].values[0]
    assert lmax_temp == lmax, f"Expected lmax {lmax}, but got {lmax_temp}"
    spin, p0, e0, x0 = df_subset[['a', 'p', 'e', 'x']].values[0]
    assert spin == a0, f"Expected spin {a0}, but got {spin}"
    assert p0 == ptemp, f"Expected p {ptemp}, but got {p0}"

    # Initialize KerrGeodesic object
    geo = KerrGeodesic(spin, p0, e0, x0, nsamples = 4)

    kappa = np.sqrt(1-spin**2)
    rp = 1 + kappa
    rm = 1 - kappa

    # Load hret data for each gauge and sum over l-modes
    hret_data_g = {}
    hret_m_g = {}
    for gauge, hret_file in zip(df_subset['gauge'], df_subset['hretlm_file']):
        hret_data_g[gauge] = np.load(os.path.join(data_dir, hret_file))
        hret_m_g[gauge] = np.sum(hret_data_g[gauge], axis=0)

    # Generate special time slices that are used to more nicely plot the asymptotic behavior
    Rstar_grid = r_grid + (1+kappa)/kappa*np.log((r_grid - rp)/2) - (1-kappa)/kappa*np.log((r_grid - rm)/2)
    Phistar_grid = spin / (2*kappa) * np.log((r_grid - rm)/(r_grid - rp))
    t_of_u_grid = 0 - Rstar_grid # u = t - Rstar
    t_of_u_grid[r_grid==p0] = 0 # this is to maintain t-slicing at r0
    t_of_v_grid = 0 + Rstar_grid # v = t + Rstar
    t_of_v_grid[r_grid==p0] = 0 # this is to maintain t-slicing at r0
    phi_grid = 0 + Phistar_grid # phi tilde = phi - Phistar

    # Prepare all of the components of the metric perturbation data by multiplying m-modes by coordinates above
    hret_inner = {}
    hret_outer = {}
    gauge_iter_dict = {'ORG': [0, 1, 2, 6, 7], # indices refer to ['ll', 'lm', 'lmbar', 'nn', 'nm', 'nmbar', 'mm', 'mbarmbar']
                       'SRG4': [0, 1, 2, 6, 7], # note that 'ln' and 'mmbar' are zero due to the gauge conditions (i.e., trace-free)
                       'ARG4': [0, 1, 2, 6, 7],
                       'IRG': [3, 5, 4, 7, 6],
                       'SRG0': [3, 5, 4, 7, 6],
                       'ARG0': [3, 5, 4, 7, 6]}
    for gauge in gauges:
        data_inner = {}
        data_outer = {}
        for i in range(3):
            data_inner[i] = np.zeros(r_grid.shape, dtype=np.complex128)
            data_outer[i] = np.zeros(r_grid.shape, dtype=np.complex128)
            for m in range(-lmax, lmax + 1):
                data_inner[i]+=hret_m_g[gauge][m,i,0]*np.exp(-1.j*geo.mode_frequency(m, 0, 0)*t_of_u_grid)*np.exp(1.j*m*phi_grid)
                data_outer[i]+=hret_m_g[gauge][m,i,1]*np.exp(-1.j*geo.mode_frequency(m, 0, 0)*t_of_v_grid)*np.exp(1.j*m*phi_grid)
            data_inner[i][data_inner[i] == 0] = None
            data_outer[i][data_outer[i] == 0] = None
        data_inner_row = [data_inner[0].copy(), data_inner[1].copy(), np.conj(data_inner[1].copy()), data_inner[2].copy(), np.conj(data_inner[2].copy())]
        data_outer_row = [data_outer[0].copy(), data_outer[1].copy(), np.conj(data_outer[1].copy()), data_outer[2].copy(), np.conj(data_outer[2].copy())]
        if gauge in gauge_iter_dict:
            gauge_iter = gauge_iter_dict[gauge]
            hret_inner[gauge] = np.zeros((8,) + data_inner_row[0].shape, dtype=np.complex128)
            hret_inner[gauge][gauge_iter] = data_inner_row
            hret_outer[gauge] = np.zeros((8,) + data_outer_row[0].shape, dtype=np.complex128)
            hret_outer[gauge][gauge_iter] = data_outer_row


    # Combine the two potentials for the SRG and ARG gauges
    hret_inner["SRG"] = [hret_inner["SRG4"][i] + hret_inner["SRG0"][i] for i in range(len(hret_inner["SRG0"]))]
    hret_outer["SRG"] = [hret_outer["SRG4"][i] + hret_outer["SRG0"][i] for i in range(len(hret_outer["SRG0"]))]
    hret_inner["ARG"] = [hret_inner["ARG4"][i] - hret_inner["ARG0"][i] for i in range(len(hret_inner["ARG0"]))]
    hret_outer["ARG"] = [hret_outer["ARG4"][i] - hret_outer["ARG0"][i] for i in range(len(hret_outer["ARG0"]))] 

    # Plot the results
    gauge_list = ['ORG', 'IRG', 'SRG', 'ARG']
    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots(4, 2, figsize=(6.5, 12), sharex=True, sharey=True)
    fig.tight_layout(pad=2.0)
    ax = ax.flatten()
    comp_list = ['$\\left|h_{ll}^\\mathrm{G,\pm}\\right|$',
                '$\\left|\mathrm{Re}[h_{lm}^\\mathrm{G,\pm}]\\right|$',
                '$\\left|\mathrm{Im}[h_{l\\bar{m}}^\\mathrm{G,\pm}]\\right|$',
                '$\\left|h_{nn}^\\mathrm{G,\pm}\\right|$',
                '$\\left|\mathrm{Re}[h_{nm}^\\mathrm{G,\pm}]\\right|$',
                '$\\left|\mathrm{Im}[h_{n\\bar{m}}^\\mathrm{G,\pm}]\\right|$',
                '$\\left|\mathrm{Re}[h_{mm}^\\mathrm{G,\pm}]\\right|$',
                '$\\left|\mathrm{Im}[h_{\\bar{m}\\bar{m}}^\\mathrm{G,\pm}]\\right|$',
                ]
    line_list = ['-', '--', '-.', ':']
    hret_inner_list = [hret_inner[gauge] for gauge in gauge_list]
    hret_outer_list = [hret_outer[gauge] for gauge in gauge_list]
    color_list = mpl.cm.viridis(np.linspace(0, 0.8, len(hret_inner_list)))


    for k in range(8):
        for j in range(4):
            data_inner = hret_inner_list[j][k]
            data_outer = hret_outer_list[j][k]
            if k == 1:
                data_inner = data_inner.real
                data_outer = data_outer.real
            elif k == 2:
                data_inner = data_inner.imag
                data_outer = data_outer.imag
            elif k == 4:
                data_inner = data_inner.real
                data_outer = data_outer.real
            elif k == 5:
                data_inner = data_inner.imag
                data_outer = data_outer.imag
            elif k == 6:
                data_inner = data_inner.real
                data_outer = data_outer.real
            elif k == 7:
                data_inner = data_inner.imag
                data_outer = data_outer.imag
            if k == 1:
                kk = 2
            elif k == 2:
                kk = 3
            elif k == 3:
                kk = 1
            else:
                kk = k
            if data_inner is not None and np.all(data_inner != 0.):
                ax[kk].plot(r_grid - rp, np.abs(data_inner), line_list[j], c=color_list[j], label = gauge_list[j])
            if data_outer is not None and np.all(data_outer != 0.):
                ax[kk].plot(r_grid - rp, np.abs(data_outer), line_list[j], c=color_list[j])
        ax[kk].plot((p0 - rp, p0 - rp), (1e-11, 1e8), 'k-', alpha = 0.2)
        ax[kk].set_ylabel(comp_list[k])
        ax[kk].set_yscale('log')
        ax[kk].set_xscale('log')
        ax[kk].set_xlim(1e-4, 2e4)
        ax[kk].set_ylim(1e-11, 1e8)
    ax[kk].set_xlabel('$(r - r_+)/M$')
    ax[kk-1].set_xlabel('$(r - r_+)/M$')
    ax[kk].legend(loc='upper left', fontsize=12)
    print(f"Saving figure for spin {spin} and p0 {p0}")
    plt.savefig(f'figures/hret_asymp_a{spin}_p{p0}.pdf', dpi=300, bbox_inches='tight')