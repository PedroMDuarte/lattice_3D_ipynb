from __future__ import division
import sympy as sym
from sympy import *


import matplotlib
matplotlib.rc('font',**{'family':'serif'})

import numpy as np

# Solution for the HTSE (High Temperature Series Expansion)

sym_z0, sym_z, sym_u, sym_B, sym_t , sym_U= symbols(r"z_{0} z u \beta t U")

m=6.
grand = -sym.log( sym_z0 ) / sym_B - sym_B * (sym_t/sym_z0)**2 * m * ( sym_z + sym_z**3 * sym_u + 2*sym_z**2 * (1-sym_u) / (sym_B*sym_U))

sym_T, sym_mu= symbols(r"T \mu")
sym_B = 1 / sym_T 
sym_z = sym.exp( sym_B*sym_mu )
sym_u = sym.exp(-sym_B*sym_U ) 

sym_z0 = 1 + 2*sym_z + sym_z**2 * sym_u
grand = -sym.log( sym_z0 ) / sym_B - sym_B * (sym_t/sym_z0)**2 * m * ( sym_z + sym_z**3 * sym_u + 2*sym_z**2 * (1-sym_u) / (sym_B*sym_U))
grand = grand.subs(sym_t,1)

Density  = -1 * diff( grand, sym_mu )
Doublons = diff( grand, sym_U )
Entropy  = -1* diff( grand, sym_T)
DensFluc = sym_T* diff( -1* diff( grand, sym_mu), sym_mu )


# Save interpolation data for the HTSE

from scipy.interpolate import CloughTocher2DInterpolator
from scipy.interpolate.interpnd import _ndim_coords_from_arrays
import matplotlib.mlab as ml

def saveHTSEInterp( Temperature ):
    mu_set = np.linspace(-60, 108, 129 )
    U_set  = np.linspace(-12, 68, 93 )
    HTSEdat = None
    for i,muval in enumerate(mu_set):
        for j,Uval in enumerate(U_set):
            dens = Density.subs(( (sym_T, Temperature), (sym_mu,muval), (sym_U,Uval) ))
            denf = DensFluc.subs(( (sym_T, Temperature), (sym_mu,muval), (sym_U,Uval) ))
            doub = Doublons.subs(( (sym_T, Temperature), (sym_mu,muval), (sym_U,Uval) ))
            entr = Entropy.subs(( (sym_T, Temperature), (sym_mu,muval), (sym_U,Uval) ))
            if "nan" in [str(dens), str(doub), str(entr)]:
                continue
            else:
                if HTSEdat is None:
                    HTSEdat = np.array([ Uval, muval, dens, denf, doub, entr ])
                else:
                    HTSEdat = np.vstack(( HTSEdat, np.array([ Uval, muval, dens, denf, doub, entr]) ))
                    
    np.savetxt('HTSEdat/HTSEinterp%03d.dat'%int(Temperature*10), HTSEdat) 
    return
  

# Save the phase diagram for a lot of temperatures 
set1 = np.hstack(( np.arange(15.6,16.4,0.8) ))
for Ti in set1:
    saveHTSEInterp( Ti )
    print Ti 
