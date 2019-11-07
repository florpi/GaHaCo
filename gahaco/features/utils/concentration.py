import numpy as np

def prada(self):
    """
    Computes the concentration as in Prada et al. (2012), instead of NFW fitting.
    (DOI: 10.1111/j.1365-2966.2012.21007.x)
    
    Returns:
            Halo concentration
    """
    scale_factor = 1.0 / (1.0 + self.snapshot.header.redshift)
    r200c_physical = self.r200c * scale_factor / 1000.0  # units Mpc

    v200 = (
        (self.snapshot.const.G * self.m200c)
        / r200c_physical
        * self.snapshot.const.Mpc ** 2
        / 1000.0 ** 2
    ) ** 0.5  # units km/s

    def y(x, vmax, v200):
        func = np.log(1 + x) - (x / (1 + x))
        return ((0.216 * x) / func) ** 0.5 - (vmax / v200)

    concentration = np.zeros((len(self.vmax)))
    for halo in range(self.N_halos):
        if v200[halo] > self.vmax[halo]:
            concentration[halo] = -9999.0
        else:
            try:
                concentration[halo] = newton(
                    y, x0=5.0, args=(self.vmax[halo], v200[halo])
                )
            except:
                concentration[halo] = -9999.0

    return concentration
