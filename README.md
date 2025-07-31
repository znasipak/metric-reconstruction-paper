# metric-reconstruction-paper
Repository of results for the paper on metric reconstruction
See the (arXiv)[https://arxiv.org/abs/2507.07746] for the latest version.

All figures from the paper are included in the `figures` directory, while numerical values of the redshift, particularly those reported in Tables I and II in the paper, are provided in the `results` directory. Specifically,
    - `z1-equatorial_comparison.csv` contains the data for the comparisons made in Table I
    - `z1-precessing.csv` contains the data listed in Table II
    - `z1-circular.csv` contains the data in the final figure, plotting the negative redshift values.

The `.csv` files are structured with the following information:
    - `gauge`: The gauge of the metric perturbation used to compute `z1`
    - `name`: Metadata concerning the computing run that generated the data
    - `a`: Dimensionless black hole spin
    - `p`: Dimensionless semilatus rectum
    - `e`: Orbital eccentricity
    - `x`: (Cosine) inclination angle with respect to the equatorial plane
    - `z0`: Geodesic redshift for frequencies geodesically related to (`a`,`p`,`e`,`x`)
    - `z1`: First-order correction for frequencies geodesically related to (`a`,`p`,`e`,`x`)
    - `z1_err`: Estimated error in `z1` based on fitting procedure
    - `lmax_cut`: Largest `lmode` included in best fit
    - `z1_lmax`: Value of `z1` data for that `lmode`

The `scripts` directory contains scripts to generate the `figures` and `results`, however they rely on a `data` directory that is too large (>15G) to store on Github. Those interested in using the raw data can reach out to the author for access.
