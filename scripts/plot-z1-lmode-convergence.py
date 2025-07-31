
## huu_l_mode_convergence_plot.py

from pybhpt.geo import KerrGeodesic
from scipy.special import ellipk
import matplotlib.pyplot as plt
import os
import json
import numpy as np
import pandas as pd
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Computer Modern"]
plt.rcParams["font.size"] = 18

def full_libration(xp):
    return np.concatenate((xp, np.flip(xp[1:-1])))

def orbit_average_gen(func, geo, axis = 0):
    th = geo.polarpoints
    r = geo.radialpoints
    thp = full_libration(th)
    rp = full_libration(r)
    q = geo.blackholespin
    sig = np.add.outer(rp**2, q**2*np.cos(thp)**2)
    if func.shape[1] != 2:
        sigA = np.array(sig)
    else:
        sigA = np.array([sig, sig])
    return np.mean(np.mean(func*sigA, axis = axis), axis = axis-1)/np.mean(sig)

def huu_reg_gen_base(q, En, Lz, Qc, r, th):
    sig = r**2 + q**2*np.cos(th)**2
    eta = pow(sig,-1)*pow(2.*(2.*r + sig)*pow(q,2)*(sig*(-1.*Qc + pow(Lz,2)) + (2.*r + sig*pow(En,2))*(-1.*sig + pow(r,2))) + pow(q,4)*pow(2.*r + sig,2) + pow(sig*(Qc + pow(Lz,2)) - (2.*r + sig*pow(En,2))*(-1.*sig + pow(r,2)),2),0.5)
    zeta = Qc - 1.*r*(2. + r*(-2. + pow(En,2))) + sig*pow(En,2) + pow(Lz,2) + pow(q,2) + 2.*r*(pow(q,2) + pow(r,2))*pow(sig,-1)
    k = 2*eta/(eta + zeta)
    K = ellipk(k)
    return 4.*K/(np.pi*np.sqrt(eta/k))

def huu_reg_gen(geo):
    En = geo.orbitalenergy
    Lz = geo.orbitalangularmomentum
    Qc = geo.carterconstant
    r = geo.radialpoints
    r0 = full_libration(r)
    th = geo.polarpoints
    th0 = full_libration(th)
    q = geo.blackholespin
    rp = np.array([r0 for th in th0]).T
    thp = np.array([th0 for r in r0])
    return huu_reg_gen_base(q, En, Lz, Qc, rp, thp)

if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "huu")
    df = pd.read_csv(os.path.join(data_dir, "huu_metadata.csv"))
    
    # Filter for cross gauge test
    df_select = df[df['name'] == 'test-5'].reset_index(drop=True)
    print(df_select)
    huu_file = {}
    params_file = {}
    huuYl_arr = {}
    params = {}

    print("Processing cross gauge test data...")
    for i in range(len(df_select)):
        huu_file[df_select.loc[i, 'gauge']] = os.path.join(df_select.loc[i, "subdir"], df_select.loc[i, "huu_file"])
        params_file[df_select.loc[i, 'gauge']] = os.path.join(df_select.loc[i, "subdir"], df_select.loc[i, "params_file"])

        huuYl_arr[df_select.loc[i, 'gauge']] = np.load(os.path.join(data_dir, huu_file[df_select.loc[i, 'gauge']]))
        with open(os.path.join(data_dir, params_file[df_select.loc[i, 'gauge']])) as json_file:
            params[df_select.loc[i, 'gauge']] = json.load(json_file)

    print("Test parameters:")
    a = params['ORG']['a']
    p = params['ORG']['p']
    e = params['ORG']['e']
    x = params['ORG']['x']
    lmax = params['ORG']["lmax"]
    nsamples = params['ORG']['nsamples']
    print(f"(a, p, e, x) = ({a}, {p}, {e}, {x})")

    geo = KerrGeodesic(a, p, e, x, nsamples = nsamples)

    lmodes = np.arange(1., lmax+1)
    huuregB = huu_reg_gen(geo)
    data = {}
    datareg = {}
    for huuYl_arr, key in [[huuYl_arr['ORG'], "ORG"], [huuYl_arr['IRG'], "IRG"], [huuYl_arr['SRG'], "SRG"]]:
        huul = huuYl_arr[0, 1:]
        data[key] = -0.5*orbit_average_gen(huul, geo, axis = 2)
        datareg[key] = -0.5*orbit_average_gen(huul - huuregB, geo, axis = 2)
        ymin = np.min(np.abs(datareg[key]))/5
        ymax = 5*np.max(np.abs(datareg[key]))

    plt.rcParams["font.size"] = 18
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(lmodes, np.abs(data['ORG']), 'o', fillstyle='none', label="$\\langle \\tilde{{z}}_1^{{\mathrm{{ORG}},\\mathrm{{rec}},\ell}}\\rangle_t$")
    for key, marker in [["ORG", "v"], ["IRG", "D"], ["SRG", "s"]]:
        ax.plot(lmodes, np.abs(datareg[key]), marker, fillstyle='none', label=f"$\\langle \\tilde{{z}}_1^{{\mathrm{{{key}}},\\mathrm{{rec}},\ell}}\\rangle_t - \\langle \\tilde{{z}}_1^{{\\mathrm{{S}}[0]}}\\rangle_t$")
    for i in range(20):
        ax.plot(lmodes, ymax*10.**(3-0.5*i)*lmodes**(-2), '--', color='gray', lw=0.5)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xticks([1, 2, 5, 10, 20])
    ax.set_xticklabels([1, 2, 5, 10, 20])
    ax.set_xlabel("$\\ell$")
    # ax.set_ylabel("$\\left|\\langle h_{uu}^{\\mathrm{R}, l} \\rangle\\right|$")
    # ax.set_title(f"$(a, p, e, x) = ({a}, {p}, {e}, {x})$")
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(1, lmax)
    ax.legend(loc='lower left', fontsize=14)
    plt.savefig('figures/z1_comp_lmodes.pdf', bbox_inches='tight', dpi=300)
