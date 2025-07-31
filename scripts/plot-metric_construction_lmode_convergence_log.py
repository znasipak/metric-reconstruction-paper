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
    lmin = 15
    lmax = 50
    gauge = 'ORG'

    # Filter the DataFrame for the specific parameters
    df_subset = df.loc[(df['gauge'] == gauge) & (df['lmax'] == lmax)]
    lmax_temp = df_subset['lmax'].values[0]
    assert lmax_temp == lmax, f"Expected lmax {lmax}, but got {lmax_temp}"

    p0_vals = np.sort(df_subset['p'].unique())
    a0_vals = np.sort(df_subset['a'].unique())
    lrange_list = np.arange(lmin, lmax + 1, 5)

    print(f"Plotting for gauge: {gauge}, lmax: {lmax}, p0 values: {p0_vals}, a0 values: {a0_vals}")

    comp_iter = 1
    kcells = 4
    fig, ax = plt.subplots(kcells, 2, figsize=(12, 15/4*kcells), sharex='col', sharey=True)
    plt.rcParams.update({'font.size': 14})

    color_list = mpl.cm.viridis(np.linspace(0, 0.9, len(lrange_list)))
    mmodes = np.arange(-lmax, lmax + 1)
    lmodes = np.arange(0, lmax + 1)

    for k in range(kcells):
        p0 = p0_vals[k]
        df_a0_0 = df_subset.loc[(df_subset['p'] == p0) & (df_subset['a'] == a0_vals[0])]
        hret_file_0 = df_a0_0['hretlm_file'].values[0]
        hret_data_g_temp_0 = np.load(os.path.join(data_dir, hret_file_0))
        df_a0_1 = df_subset.loc[(df_subset['p'] == p0) & (df_subset['a'] == a0_vals[1])]
        hret_file_1 = df_a0_1['hretlm_file'].values[0]
        hret_data_g_temp_1 = np.load(os.path.join(data_dir, hret_file_1))
        
        geo0 = KerrGeodesic(a0_vals[0], p0, 0., 1., nsamples = 4)
        geo1 = KerrGeodesic(a0_vals[1], p0, 0., 1., nsamples = 4)
        
        kappa0 = np.sqrt(1 - geo0.blackholespin**2)
        kappa1 = np.sqrt(1 - geo1.blackholespin**2)
        rp0 = 1 + kappa0
        rm0 = 1 - kappa0
        rp1 = 1 + kappa1
        rm1 = 1 - kappa1

        r_grid_file_0 = df_a0_0['rgrid_file'].values[0]
        r_grid0 = np.load(os.path.join(data_dir, r_grid_file_0))
        rstar0 = r_grid0 + (1+kappa0)/kappa0*np.log((r_grid0 - rp0)/2) - (1-kappa0)/kappa0*np.log((r_grid0 - rm0)/2)
        rstar0p = p0 + (1+kappa0)/kappa0*np.log((p0 - rp0)/2) - (1-kappa0)/kappa0*np.log((p0 - rm0)/2)
        tgrid0 = rstar0 - 0
        tgrid0[r_grid0 == p0] = 0

        phistar0 = geo0.blackholespin / (2*kappa0) * np.log((r_grid0 - rm0)/(r_grid0 - rp0))
        phistar0p = geo0.blackholespin / (2*kappa0) * np.log((p0 - rm0)/(p0 - rp0))
        phistar0p = 0
        phigrid0 = phistar0 - phistar0p
        phigrid0[r_grid0 == p0] = 0

        r_grid_file_1 = df_a0_1['rgrid_file'].values[0]
        r_grid1 = np.load(os.path.join(data_dir, r_grid_file_1))
        rstar1 = r_grid1 + (1+kappa1)/kappa1*np.log((r_grid1 - rp1)/2) - (1-kappa1)/kappa1*np.log((r_grid1 - rm1)/2)
        rstar1p = p0 + (1+kappa1)/kappa1*np.log((p0 - rp1)/2) - (1-kappa1)/kappa1*np.log((p0 - rm1)/2)
        tgrid1 = rstar1 - 0
        tgrid1[r_grid1 == p0] = 0

        phistar1 = geo1.blackholespin / (2*kappa1) * np.log((r_grid1 - rm1)/(r_grid1 - rp1))
        phistar1p = geo1.blackholespin / (2*kappa1) * np.log((p0 - rm1)/(p0 - rp1))
        phistar1p = 0
        phigrid1 = phistar1 - phistar1p
        phigrid1[r_grid1 == p0] = 0
        
        for i in [0,1]:
            j = 0

            if i == 0:
                uv_sgn = 1
            else:
                uv_sgn = -1
            hretlm0 = np.array([hret_data_g_temp_0[:, mm]*np.exp(uv_sgn*1.j * geo0.mode_frequency(mm, 0, 0) * tgrid0)*np.exp(1j*mm*phigrid0) for mm in mmodes]).swapaxes(0, 1)
            hretlm1 = np.array([hret_data_g_temp_1[:, mm]*np.exp(uv_sgn*1.j * geo1.mode_frequency(mm, 0, 0) * tgrid1)*np.exp(1j*mm*phigrid1) for mm in mmodes]).swapaxes(0, 1)
            rlower0 = rp0
            rlower1 = rp1
            for ll in lrange_list:
                grid_lmax_val = np.sum(hretlm0, axis = 1)[ll, comp_iter, i]
                grid_lmax_val[np.abs(grid_lmax_val) == 0] = None
                grid_max_val = np.sum(np.sum(hretlm0, axis = 1)[:, comp_iter, i], axis = 0)
                grid_max_val[np.abs(grid_lmax_val) == 0] = None
                if i == 0:
                    label = f'$\\ell_\\mathrm{{max}}={ll}$'
                else:
                    label = None
                ax[k, 0].plot(r_grid0 - rlower0, np.abs(grid_lmax_val/grid_max_val), c=color_list[j], label = label)
                
                grid_lmax_val = np.sum(hretlm1, axis = 1)[ll, comp_iter, i]
                grid_lmax_val[np.abs(grid_lmax_val) == 0] = None
                grid_max_val = np.sum(np.sum(hretlm1, axis = 1)[:, comp_iter, i], axis = 0)
                grid_max_val[np.abs(grid_lmax_val) == 0] = None
                if i == 0:
                    label = f'$\\ell_\\mathrm{{max}}={ll}$'
                else:
                    label = None
                ax[k, 1].plot(r_grid1 - rlower1, np.abs(grid_lmax_val/grid_max_val), c=color_list[j], label = label)

                ax[k, 0].plot((p0 - rlower0, p0 - rlower0), (1e-25, 1), '-', c='grey', lw = 0.5)
                ax[k, 1].plot((p0 - rlower1, p0 - rlower1), (1e-25, 1), '-', c='grey', lw = 0.5)
                
                xmin0 = r_grid0[0] - rlower0 + 1e-4
                xmin1 = r_grid1[0] - rlower1 + 1e-4
                xmax = 5000

                ax[k, 0].set_yscale('log')
                ax[k, 0].set_xscale('log')
                ax[k, 0].set_xlim(xmin0, xmax)
                ax[k, 0].set_ylim(1e-25, 1)

                ax[k, 1].set_yscale('log')
                ax[k, 1].set_xscale('log')
                ax[k, 1].set_xlim(xmin1, xmax)
                ax[k, 1].set_ylim(1e-25, 1)
                ax[k, 1].text(2*(p0 - rlower1), 1e-2, f"$p = {int(p0)}$", fontsize=13, color = 'grey', usetex = True)

                ax[k, 0].set_ylabel(f"$\\hat{{\\Delta}}^{{\\mathrm{{{gauge}}},\\ell_\\mathrm{{max}}}}_{{lm}}$")
                j += 1
    ax[0, 0].legend(loc='lower right', fontsize=11, ncol = 2)
    ax[0,0].set_title('$a = 0.6$', fontsize=13)
    ax[0,1].set_title('$a = 0.9$', fontsize=13)
    ax[-1, 0].set_xlabel('$r/M$')
    ax[-1, 1].set_xlabel('$r/M$')
    plt.savefig(f"figures/lmode-contribution-log-{gauge}.pdf", dpi=300, bbox_inches='tight', format='pdf')