# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from IPython.display import display

from sympy.interactive import printing
printing.init_printing()

from __future__ import division
import sympy as sym
from sympy import *


import matplotlib
matplotlib.rc('font',**{'family':'serif'})

# <markdowncell>

# # Outline of the HTSE solution to second order

# <codecell>

z0, z, u, B, t , U= symbols(r"z_{0} z u \beta t U")
m=6.
grand = -sym.log( z0 ) / B - B * (t/z0)**2 * m * ( z + z**3 * u + 2*z**2 * (1-u) / (B*U))
grand

# <codecell>

T, mu= symbols(r"T \mu")
B = 1 / T 
z = sym.exp( B*mu )
u = sym.exp(-B*U ) 

z0 = 1 + 2*z + z**2 * u
grand = -sym.log( z0 ) / B - B * (t/z0)**2 * m * ( z + z**3 * u + 2*z**2 * (1-u) / (B*U))
grand

# <codecell>

grand = grand.subs(t,1)
grand 

# <codecell>

sym.simplify(grand)

# <codecell>

Density = -1 * diff( grand, mu )
Density

# <codecell>

Doublons = diff( grand, U )
Doublons

# <codecell>

Entropy = -1* diff( grand, T)
Entropy

# <codecell>

DensFluc = T* diff( -1* diff( grand, mu), mu )
DensFluc

# <markdowncell>

# # Define interpolation functions for the HTSE and the Fuchs phase diagrams.  

# <markdowncell>

# ## Solution, interpolation, and plots for HTSE

# <codecell>

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
    mu_set = np.linspace(-12, 48, 83 )
    U_set  = np.linspace(-12, 48, 83 )
    
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
   

# Load interpolation data and make interpolations and plots 
    
def getHTSEPoints( Temperature, name="density"):
    try:
        HTSE = np.loadtxt('HTSEdat/HTSEinterp%03d.dat'%int(Temperature*10))
    except:
        saveHTSEInterp( Temperature )
        HTSE = np.loadtxt('HTSEdat/HTSEinterp%03d.dat'%int(Temperature*10))
    namedict = {"density":2, "densfluc":3, "doublons":4, "entropy":5}

    U_   = HTSE[:,0]
    mu_  = HTSE[:,1]
    qty_ = HTSE[:, namedict[name]]
    
    return U_, mu_, qty_

def getHTSEInterp( Temperature, name='density'):
    U_, mu_, qty_ = getHTSEPoints( Temperature, name=name)
    points = _ndim_coords_from_arrays((U_, mu_))
    return CloughTocher2DInterpolator(points, qty_)

def getHTSEGridDat( Temperature, name='density'):
    U_, mu_, qty_ = getHTSEPoints( Temperature, name=name)
    xi = np.linspace( U_.min(), U_.max(), 300)
    yi = np.linspace( mu_.min(), mu_.max(), 300)
    zq = ml.griddata(U_, mu_, qty_, xi,yi)
    return mu_, U_, zq, xi, yi, qty_

def makeHTSEPlot( ax, Temperature, name='density' ):
    mu_, U_, zn, xi, yi, zi = getHTSEGridDat( Temperature, name=name)
    titleDict = {'density':'Atoms per site', 
                 'doublons':'Double occupancy', 
                 'entropy':'Entropy per site',
                 'densfluc':'Density fluctuations'}
    contourDict = {'density':[1.9,1.5,1.1,1.0,0.50,0.1],
                  'doublons':[0.4,0.30,0.20,0.10,0.02,0.98,0.90,0.50],
                  'entropy':[0.2, 0.3,0.46, np.log(2), np.log(3), np.log(4), 0.75, 1.0],
                  'densfluc':[0.05, 0.20, 0.4, 0.8]}
    
    ax.set_title(titleDict[name])
    c0  =ax.contour(xi, yi, zn, contourDict[name], linewidths = 0.5, colors = 'k')
    plt.clabel(c0, inline=1, fontsize=10)
    im0 =ax.pcolormesh(xi, yi, zn, cmap = plt.get_cmap('rainbow'))
    #plt.scatter(x, y, marker = 'o', c = 'b', s = 5, zorder = 10)
    plt.axes( ax)
    plt.colorbar(im0) 
    
def HTSEPhaseDiagram( Temperature = 2.4 ):
    fig = plt.figure(figsize=(11,8))
    gs = matplotlib.gridspec.GridSpec( 2,2, wspace=0.2)

    ax0 = fig.add_subplot( gs[0,0])
    ax1 = fig.add_subplot( gs[0,1])
    ax2 = fig.add_subplot( gs[1,0])
    ax3 = fig.add_subplot( gs[1,1])

    for ax in [ax0,ax1,ax3]:
        ax.set_xlabel("$U/t$",fontsize=16)
        ax.set_ylabel("$\mu/t$",fontsize=16,rotation=0,labelpad=-5)
        #ax.set_xlim( -12., 48.)
        #ax.set_ylim( -12., 48.)
        #ax.grid()
        
    makeHTSEPlot( ax0, Temperature, name='density' )
    makeHTSEPlot( ax1, Temperature, name='doublons' )
    makeHTSEPlot( ax2, Temperature, name='densfluc' )
    makeHTSEPlot( ax3, Temperature, name='entropy' )

    gs.tight_layout(fig, rect=[0.,0.,1.0,1.0])
    fig.savefig('HighT_figures/HTSE_phasesT%03d.png'%(10*Temperature),dpi=180)

# <codecell>

HTSEPhaseDiagram( Temperature = 2.5 )

# <markdowncell>

# ## Functions to interpolate and make plots of Fuchs phase diagram

# <codecell>

doub = {}
entr = {}
enrg = {}
for U in [4, 6, 8, 10, 12]:
    doub[U] = np.loadtxt("FuchsThermodynamics/tables/doubleocc_U%d.dat"%U)
    entr[U] = np.loadtxt("FuchsThermodynamics/tables/entropy_U%d.dat"%U)
    enrg[U] = np.loadtxt("FuchsThermodynamics/tables/energy_U%d.dat"%U)
    
fuchs ={}
fuchs['doublons'] = doub
fuchs['entropy']  = entr
fuchs['energy']   = enrg

from scipy.interpolate import CloughTocher2DInterpolator
from scipy.interpolate.interpnd import _ndim_coords_from_arrays
import matplotlib.mlab as ml

def getFuchsPoints( Temperature, name='density', Uset = [4,6,8,10,12]):
    if name == 'density':
        qtydict = fuchs['doublons']
        qd = {'mu':1, 'qty':2}
    else:
        qtydict = fuchs[name]
        qd = {'mu':1, 'qty':4}
    qty = []
    
    
    for U in Uset:
        if name == 'doublons':
            for row in qtydict[U]:
                if row[0] == Temperature:
                    if row[qd['mu']] == 0.:
                        half_fill_doublons = row[qd['qty']]
                    
        for row in qtydict[U]:
            if row[0] == Temperature:
                qty.append(np.array([ row[qd['mu']] + U/2. ,  float(U), row[qd['qty']] ]))
                
                # Mirror quantitites to get positive chemical potential
                # We use the particle-hole symmetry of the Hubbard hamiltonian
                # (http://quest.ucdavis.edu/tutorial/hubbard7.pdf)
                # The entropy, density, and local moment are symmetric about mu=U/2
                # The double occupancy is given by d = (n-m^2)/2  where 
                # n is the density and m^2 is the local moment.  
                if name == 'density':
                    mirrored = 2. - row[qd['qty']]
                if name == 'entropy':
                    mirrored = row[qd['qty']]
                    
                if name == 'doublons':
                    density = row[2]
                    mirroredDensity = 2. - density
                    localMoment = density - 2 * row[qd['qty']] 
                    mirrored = ( mirroredDensity - localMoment  ) / 2.
                qty.append(np.array([ -1.*row[qd['mu']] +U/2.,  float(U), mirrored ]))
                
    qty = np.array(qty)
    #Sort the points by col0, which is chemical potential
    idx = np.argsort(qty[:,0])
    mu_ = qty[ idx, 0]
    U_  = qty[ idx, 1]
    qty_= qty[ idx, 2]
    
    return U_, mu_, qty_

def getFuchsInterp( Temperature, name='density', Uset = [4,6,8,10,12]):
    U_, mu_, qty_ = getFuchsPoints( Temperature, name=name, Uset = Uset)
    points = _ndim_coords_from_arrays((U_, mu_))
    if len ( Uset) > 1:
        return CloughTocher2DInterpolator(points, qty_)
    else:
        return None

def getFuchsGridDat( Temperature, name='density', Uset = [4,6,8,10,12]):
    U_, mu_, qty_ = getFuchsPoints( Temperature, name=name, Uset = Uset)
    xi = np.linspace( U_.min(), U_.max(), 300)
    yi = np.linspace( mu_.min(), mu_.max(), 300)
    if len ( Uset) > 1:
        zq = ml.griddata(U_, mu_, qty_, xi,yi)
    else:
        zq = None
    return mu_, U_, zq, xi, yi, qty_
    
def makeFuchsPlot( ax, Temperature, name='density' ):
    mu_, U_, zn, xi, yi, zi = getFuchsGridDat( Temperature, name=name)
    titleDict = {'density':'Atoms per site', 
                 'doublons':'Double occupancy', 
                 'entropy':'Entropy per site'}
    contourDict = {'density':[1.9,1.5,1.1,1.0,0.50,0.1],
                  'doublons':[0.4,0.30,0.20,0.10,0.02],
                  'entropy':[0.2, 0.3,0.46, np.log(2), np.log(3), np.log(4)]}
    
    ax.set_title(titleDict[name])
    c0  =ax.contour(xi, yi, zn, contourDict[name], linewidths = 0.5, colors = 'k')
    plt.clabel(c0, inline=1, fontsize=10)
    im0 =ax.pcolormesh(xi, yi, zn, cmap = plt.get_cmap('rainbow'))
    #plt.scatter(x, y, marker = 'o', c = 'b', s = 5, zorder = 10)
    plt.axes( ax)
    plt.colorbar(im0) 
    

    
def FuchsPhaseDiagram( Temperature = 1. ):
    fig = plt.figure(figsize=(11,8))
    gs = matplotlib.gridspec.GridSpec( 2,2, wspace=0.2)

    ax0 = fig.add_subplot( gs[0,0])
    ax1 = fig.add_subplot( gs[0,1])
    #ax2 = fig.add_subplot( gs[1,0])
    ax3 = fig.add_subplot( gs[1,1])

    for ax in [ax0,ax1,ax3]:
        ax.set_xlabel("$U/t$",fontsize=16)
        ax.set_ylabel("$\mu/t$",fontsize=16,rotation=0,labelpad=-5)
        #ax.set_xlim( -12., 48.)
        #ax.set_ylim( -12., 48.)
        #ax.grid()
        
    makeFuchsPlot( ax0, Temperature, name='density' )
    makeFuchsPlot( ax1, Temperature, name='doublons' )
    makeFuchsPlot( ax3, Temperature, name='entropy' )

    gs.tight_layout(fig, rect=[0.,0.,1.0,1.0])
    fig.savefig('HighT_figures/FUCHS_phasesT%03d.png'%(10*Temperature),dpi=180)

# <codecell>

FuchsPhaseDiagram( Temperature = 0.4 )

# <markdowncell>

# ## Simple examples of usage for the interpolation functions:

# <markdowncell>

# Evaluate the thermodynamic quantities at some value of $U\ $ and $\mu\ $

# <codecell>

Temperature = 2.4
fdens = getFuchsInterp( Temperature, name="density")
fdoub = getFuchsInterp( Temperature, name="doublons")
fentr = getFuchsInterp( Temperature, name="entropy")

fHdens = getHTSEInterp( Temperature, name="density")
fHdoub = getHTSEInterp( Temperature, name="doublons")
fHentr = getHTSEInterp( Temperature, name="entropy")

print "Temperature = {0:4,.2f}".format(Temperature)
Uval = 4. ; muval = 10. 
print "U/t = {0:4,.2f}".format(Uval)
print "mu  = {0:4,.2f}\n".format(muval)
print "Method   {0:>10s}{1:>10s}".format( "Fuchs", "HTSE")
print "-"*40
print "Density  {0:10,.2f}{1:10,.2f}".format( float(fdens(Uval, muval)), float(fHdens(Uval, muval)) )
print "Doublons {0:10,.2f}{1:10,.2f}".format( float(fdoub(Uval, muval)), float(fHdoub(Uval, muval)) )
print "Entropy  {0:10,.2f}{1:10,.2f}".format( float(fentr(Uval, muval)), float(fHentr(Uval, muval)) )                                                    



# <markdowncell>

# Example to make a chemical potential cut at fixed U, starting from a filling of 1 particle per site.   Compare with Fig 5. in Fuchs et. al. PRL 106, 030401 (2011)

# <codecell>

def trap_fixed_U( U ):
    mu_ = np.linspace( -20, U/2, 100)
    U_  = np.ones_like( mu_ ) * U
    return np.transpose(np.vstack(( U_, mu_)))

for Temp in [0.4, 0.5, 0.6, 1.0, 2.0, 4.0 ]:
    fentr = getFuchsInterp( Temp, name="entropy")
    xy = trap_fixed_U( 8. )
    z = fentr(xy)
    plt.plot( xy[:,1], z ) 

# <markdowncell>

# Do the same chemical potential cut for the HTSE phase diagram

# <codecell>

for Temp in [1.6, 2.6, 4.6, 10.6 ]:
    fentr = getHTSEInterp( Temp, name="entropy")
    xy = trap_fixed_U( 8. )
    z = fentr(xy)
    plt.plot( xy[:,1], z ) 

# <markdowncell>

# Compare the Fuchs and HTSE phase diagrams.  They should agree at high temperatures

# <codecell>

def mu_cut( U ):
    mu_ = np.linspace( -20, 20, 100)
    U_  = np.ones_like( mu_ ) * U
    return np.transpose(np.vstack(( U_, mu_)))

def plotCompare( ax, quantity ):
    for Temp in [0.4, 0.5, 0.6, 1.0, 1.6, 2.6, 4.6 ]:
        fq = getFuchsInterp( Temp, name=quantity)
        fHq = getHTSEInterp( Temp, name=quantity)
        xy = mu_cut( 8. )
        z  = fq(xy)
        zH = fHq(xy)
        FuchsLine, = ax.plot( xy[:,1], z ,alpha=.5, lw=2, label = "$T/t=%.1f$"%Temp) 
        if Temp > 1.0:
            ax.plot( xy[:,1], zH, ls='--',lw=2, color=FuchsLine.get_color())
    
fig = plt.figure(figsize=(13,4.5))
gs = matplotlib.gridspec.GridSpec( 1,3, wspace=0.2)

ax0 = fig.add_subplot( gs[0,0])
ax1 = fig.add_subplot( gs[0,1])
ax2 = fig.add_subplot( gs[0,2])

ax0.set_ylabel("Atoms per site",fontsize=14,rotation=90,labelpad=0)
ax1.set_ylabel("Double occupancy",fontsize=14,rotation=90,labelpad=0)
ax2.set_ylabel("Entropy",fontsize=14,rotation=90,labelpad=0)

for ax in [ax0,ax1,ax2]:
    ax.set_xlabel("$\mu/t$",fontsize=16)
    #ax.set_xlim( -12., 48.)
    #ax.set_ylim( -12., 48.)
    ax.grid()
    
plotCompare(ax0, "density")
plotCompare(ax1, "doublons")
plotCompare(ax2, "entropy")

fig.suptitle('$U/t = 8$', fontsize=20)
ax2.legend( bbox_to_anchor=(1.03,1.00), \
            loc='upper left', numpoints=1, \
             prop={'size':12}, handlelength=1.1, handletextpad=0.5 )

gs.tight_layout(fig, rect=[0.,0.,0.8,0.9])
fig.savefig('HighT_figures/FUchsHTSE_Compare.png',dpi=180)

# <markdowncell>

# Plot the Fuchs phase diagram entropy as a function of U, at half-filling

# <codecell>

def half_filling( U ):
    mu = U/2
    return np.transpose(np.vstack(( U, mu)))

U = np.linspace( 4., 12., 100)
for Temp in [0.4, 0.5, 0.6, 1.0, 2.0, 4.0 ]:
    fentr = getFuchsInterp( Temp, name="entropy")
    xy = half_filling( U )
    z = fentr(xy)
    plt.plot( xy[:,0], z ) 

#plt.plot( U, fentr( half_filling(U) ))
#plt.plot( U, fHentr(half_filling(U) ))
#plt.plot( U, fdens( half_filling(U) ))

