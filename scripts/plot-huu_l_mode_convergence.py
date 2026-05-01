
## huu_l_mode_convergence_plot.py

from pybhpt.geo import KerrGeodesic
from scipy.special import ellipk
import matplotlib.pyplot as plt
import os
import json
import numpy as np
import pandas as pd
from pybhpt.geo import (kerrgeo_Vt_radial, 
                        kerrgeo_Vt_polar)

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

def ut_gen(q, En, Lz, Qc, r, th):
    Sigma = np.add.outer(r**2, q**2*np.cos(th)**2)
    utr = np.array([kerrgeo_Vt_radial(q, En, Lz, Qc, ri) for ri in r])
    utth = np.array([kerrgeo_Vt_polar(q, En, Lz, Qc, thi) for thi in th])
    ut_rth = np.add.outer(utr, utth)
    return (ut_rth)/Sigma

def z0_gen_geo(geo):
    En = geo.orbitalenergy
    Lz = geo.orbitalangularmomentum
    Qc = geo.carterconstant
    r = full_libration(geo.radialpoints)
    th = full_libration(geo.polarpoints)
    q = geo.blackholespin

    Sigma = np.add.outer(r**2, q**2*np.cos(th)**2)
    return np.mean(Sigma)/np.mean(ut_gen(q, En, Lz, Qc, r, th)*Sigma)

if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "huu")
    df = pd.read_csv(os.path.join(data_dir, "huu_metadata.csv"))
    
    # Filter for e = 0.6
    df_highe = df[df['e'] == 0.6]
    huu_file = os.path.join(df_highe["subdir"].values[0], df_highe["huu_file"].values[0])
    params_file = os.path.join(df_highe["subdir"].values[0], df_highe["params_file"].values[0])

    huuYl_arr = np.load(os.path.join(data_dir, huu_file))
    with open(os.path.join(data_dir, params_file)) as json_file:
        params = json.load(json_file)

    a = params['a']
    p = params['p']
    e = params['e']
    x = params['x']
    
    print(a,p,e,x)
    print(params["nsamples"])
    
    geo = KerrGeodesic(a, p, e, x, nsamples = params["nsamples"])
    z0 = z0_gen_geo(geo)

    plt.rcParams["font.size"] = 18
    fig, ax = plt.subplots(figsize=(6, 6))
    data0 = 0.5*z0*np.abs(orbit_average_gen(huuYl_arr[0] - huu_reg_gen(geo), geo, axis = 2))
    data1 = 0.5*z0*np.abs(orbit_average_gen(huuYl_arr[1] - huu_reg_gen(geo), geo, axis = 2))
    data2 = (data0 + data1)/2
    l0_list = np.arange(0., len(data0))
    lmax = l0_list[-1]
    lmax = 16
    ymin = np.min(np.abs(data0[:lmax]))/5
    ymax = 5*np.max(np.abs(data1[:lmax]))
    ax.plot(l0_list, data0, 'o', fillstyle='none', label="$r \\rightarrow r_p^-$")
    ax.plot(l0_list, data1, 'd', fillstyle='none', label="$r \\rightarrow r_p^+$")
    ax.plot(l0_list, data2, 'x', fillstyle='none', label="average")
    for i in range(25):
        ax.plot(l0_list[1:], ymax*10.**(3-0.5*i)*l0_list[1:]**(-2), '--', color='gray', lw=0.5)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xticks([1, 2, 5, 10, 20])
    ax.set_xticklabels([1, 2, 5, 10, 20])
    ax.set_xlabel("$\\ell$")
    ax.set_ylabel("$\\left|\\langle \\tilde{{z}}_1^{{\mathrm{{ORG}},\\mathrm{{rec}},\ell}}\\rangle_t - \\langle \\tilde{{z}}_1^{{\\mathrm{{S}}[0]}}\\rangle_t\\right|$")
    # ax.set_title(f"$(a, p, e, x) = ({a}, {p:0.4f}, {e}, {x})$")
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(1, lmax)
    ax.legend(loc='upper right', fontsize=12)
    plt.savefig(f"figures/huuYl-plot-{a}-{p:0.4f}-{e}-{x}.pdf", bbox_inches='tight')
