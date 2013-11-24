# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from vec3 import vec3, cross
import scipy.constants as C 

def beam(xb,yb,zb,wx,wy,wavelen):
    zRx = np.pi * wx**2 / wavelen
    zRy = np.pi * wy**2 / wavelen 
    
    sqrtX = np.sqrt( 1 + np.power(zb/zRx,2) ) 
    sqrtY = np.sqrt( 1 + np.power(zb/zRy,2) ) 
    return np.exp( -2.*( np.power(xb/(wx*sqrtX ),2) + np.power(yb/(wy*sqrtY),2) )) / sqrtX / sqrtY

def U2uK( wavelen ):
    Cc = C.c * 1e6 # speed of light um s^-1
    Gamma = 2*np.pi *5.9e6 # linewidth s^-1
    
    omega0  = 2*np.pi*Cc / .671
    omegaL  = 2*np.pi*Cc / wavelen
    intensity = 1.0 
    depthJ  = (intensity)* -3*np.pi* Cc**2*Gamma / ( 2*omega0**3) * ( 1/(omega0 - omegaL )  + 1/(omega0 + omegaL ) ) # Joule
    depthuK = depthJ / C.k  *1e6 # C.k is Boltzmann's constant
    return depthuK

def Erecoil( wavelen, mass):
    inJ  = C.h**2 / ( 2* mass*C.physical_constants['unified atomic mass unit'][0] * (wavelen*1e-6)**2 ) 
    inuK = inJ / C.k *1e6
    return inuK
    
                  

class GaussBeam:
    """Properties of a gaussian beam""" 
    def __init__( self,
                 **kwargs ):
        self.mW = kwargs.get('mW',1000.0 )
        self.w  = kwargs.get('waists', (30.,30.) )
        self.l  = kwargs.get('wavelength', 1.064 )
        #
        self.axis   = kwargs.get('axis', (np.pi/2,0.) )
        self.origin = kwargs.get('origin', vec3(0,0,0) )
        
        
        # Make sure vectors are of type(vec3)
        self.axisvec = vec3()
        th = self.axis[0]
        ph = self.axis[1]
        self.axisvec.set_spherical( 1, th, ph) 
        self.origin = vec3(self.origin)
        
        # Calculate two orthogonal directions 
        # which will be used to specify the beam waists
        self.orth0 = vec3( np.cos(th)*np.cos(ph) , np.cos(th)*np.sin(ph), -1.*np.sin(th) )
        self.orth1 = vec3( -1.*np.sin(ph), np.cos(ph), 0. )
        
    def transform(self, X, Y, Z):
        # coordinates into beam coordinates 
        zb = X*self.axisvec[0] + Y*self.axisvec[1] + Z*self.axisvec[2]
        xb = X*self.orth0[0]   + Y*self.orth0[1]   + Z*self.orth0[2]
        yb = X*self.orth1[0]   + Y*self.orth1[1]   + Z*self.orth1[2]
        return xb,yb,zb
        
    def __call__( self, X, Y, Z):
        xb,yb,zb = self.transform( X,Y,Z)
        
        gauss = beam( xb,yb,zb, self.w[0], self.w[1], self.l)
        intensity = (2/np.pi)* self.mW/1000. /self.w[0]/self.w[1] *gauss  # W um^-2
        
        return U2uK(self.l)*intensity
        
class LatticeBeam(GaussBeam):
    def __init__(self, **kwargs):
        """Lattice beam, with retro factor and polarization """
        GaussBeam.__init__(self, **kwargs)
        self.scale = kwargs.get('scale',10.)
        self.mass  = kwargs.get('mass', 6.0)
        self.Er    = kwargs.get('Er', 7.0)
        self.retro = kwargs.get('retro', 1.0)
        self.alpha = kwargs.get('alpha', 1.0)
        self.Er0  = Erecoil( self.l , self.mass)  
        self.mW = 1000*(self.Er/4.)*self.Er0 / np.abs( U2uK(self.l)*2/np.pi )  * self.w[0]*self.w[1] 
        
        
    def __call__( self, X, Y, Z):
        xb,yb,zb = self.transform( X,Y,Z)
        
        gauss = beam( xb,yb,zb, self.w[0], self.w[1], self.l)
        intensity = (2/np.pi)* self.mW/1000. /self.w[0]/self.w[1] *gauss  # W um^-2
        
        lattice =  np.power(np.cos(2*np.pi/self.l * zb/self.scale ),2)*4*np.sqrt(self.retro*self.alpha)\
                 + ( 1 + self.retro - 2*np.sqrt(self.retro*self.alpha) )
        
        return U2uK(self.l)*intensity*lattice
    
    def getBottom( self, X, Y, Z):
        xb,yb,zb = self.transform( X,Y,Z)
        
        gauss = beam( xb,yb,zb, self.w[0], self.w[1], self.l)
        intensity = (2/np.pi)* self.mW/1000. /self.w[0]/self.w[1] *gauss  # W um^-2
        
        latticeBot = 4*np.sqrt(self.retro*self.alpha)  + 1 + self.retro - 2*np.sqrt(self.retro*self.alpha)
        return U2uK(self.l)*intensity * latticeBot
    
    def getV0( self, X, Y, Z):
        xb,yb,zb = self.transform( X,Y,Z)
        
        gauss = beam( xb,yb,zb, self.w[0], self.w[1], self.l)
        intensity = (2/np.pi)* self.mW/1000. /self.w[0]/self.w[1] *gauss  # W um^-2
    
        latticeV0  = 4*np.sqrt(self.retro*self.alpha) 
        return np.abs(U2uK(self.l)*intensity * latticeV0)

        

# <codecell>

lattTri = potential([ LatticeBeam( axis=(np.pi/2,0) ),
                     LatticeBeam( axis=(np.pi/2,2*np.pi*(1./3.)) ),
                     LatticeBeam( axis=(np.pi/2,2*np.pi*(2./3.)) )], units=('$E_{R}$', 1/Erecoil(1.064,6)) )
lattTri.plot3Cross()

# <codecell>

latt3d = potential([ LatticeBeam( axis=(np.pi/2,0) ),
                     LatticeBeam( axis=(np.pi/2,np.pi/2) ),
                     LatticeBeam( axis=(0,0) )], units=('$E_{R}$', 1/Erecoil(1.064,6)) )
#latt3d.plot3Line()
#latt3d.plotCross( normal=(0.,0.), lims0=(-50,50), lims1=(-50,50), npoints=240)
latt3d.plot3Cross()

# <codecell>

odt = potential([ GaussBeam(mW=38000., waists=(20.,20.), axis=(np.pi/2,30.*np.pi/180.)),
                  GaussBeam(mW=38000., waists=(20.,20.), axis=(np.pi/2,15.*np.pi/180.)) ])
#odt.plotLine()
#odt.plotCross(normal=(0.,0.), lims0=(-500,500),lims1=(-500,500),npoints=120)
odt.plot3Cross(extents=100)

    

# <codecell>

from mpl_toolkits.mplot3d import axes3d

class potential:
    def __init__(self, beams, **kwargs ):
        self.units = kwargs.get('units', ('$\mu\mathrm{K}$', 1.))
        self.unitlabel = self.units[0]
        self.unitfactor = self.units[1] 
        self.beams = beams 
        
    def evalpotential( self, X, Y, Z):
        EVAL = np.zeros_like(X) 
        for b in self.beams:
            EVAL += b(X,Y,Z)
        return EVAL* self.unitfactor
        
    def plotLine(self, **kwargs):
        gs      = kwargs.get('gs',None)
        figGS   = kwargs.get('figGS',None)
        func    = kwargs.get('func',None)
        origin  = kwargs.get('origin', vec3(0.,0.,0.))
        direc   = kwargs.get('direc', (np.pi/2, 0.))
        lims    = kwargs.get('lims', (-80.,80.))
        extents = kwargs.get('extents',None)
        npoints = kwargs.get('npoints', 320)
        
        if func is None:
            func = self.evalpotential
        
        if extents is not None:
            lims = (-extents, extents)
        
        # Prepare set of points for plot 
        t = np.linspace( lims[0], lims[1], npoints )
        unit = vec3()
        th = direc[0]
        ph = direc[1] 
        unit.set_spherical(1, th, ph) 
        # Convert vec3s to ndarray
        unit = np.array(unit)
        origin = np.array(origin) 
        #
        XYZ = origin + np.outer(t, unit)
        X = XYZ[:,0]
        Y = XYZ[:,1]
        Z = XYZ[:,2] 
        
        if gs is None:
            # Prepare figure
            fig = plt.figure(figsize=(9,4.5))
            gs = matplotlib.gridspec.GridSpec( 1,2, wspace=0.2)
        
            ax0 = fig.add_subplot( gs[0,0], projection='3d')
            ax1 = fig.add_subplot( gs[0,1])
        else:
            gsSub0 = matplotlib.gridspec.GridSpecFromSubplotSpec( 2,2, subplot_spec=gs,\
                         width_ratios=[1,1.6], height_ratios=[1,1],\
                         wspace=0.25)
            ax0 = figGS.add_subplot( gsSub0[0,0], projection='3d')
            ax1 = figGS.add_subplot( gsSub0[0:2,1] )
            ax2 = figGS.add_subplot( gsSub0[1,0] )
            
            ax1.set_xlim( lims[0],lims[1])
            ax2.set_xlim( lims[0]/2.,lims[1]/2.)
            ax2.grid()
            ax2.set_xlabel('$\mu\mathrm{m}$', fontsize=14)
            
            
        ax0.plot(X, Y, Z, c='blue', lw=3)
        ax0.set_xlabel('X')
        ax0.set_ylabel('Y')
        ax0.set_zlabel('Z')
        lmin = min([ ax0.get_xlim()[0], ax0.get_ylim()[0], ax0.get_zlim()[0] ] )
        lmax = max([ ax0.get_xlim()[1], ax0.get_ylim()[1], ax0.get_zlim()[1] ] )
        ax0.set_xlim( lmin, lmax )
        ax0.set_ylim( lmin, lmax )
        ax0.set_zlim( lmin, lmax )
        LMIN = np.ones_like(X)*lmin
        LMAX = np.ones_like(X)*lmax
        ax0.plot(X, Y, LMIN, c='gray', lw=2,alpha=0.3)
        ax0.plot(LMIN, Y, Z, c='gray', lw=2,alpha=0.3)
        ax0.plot(X, LMAX, Z, c='gray', lw=2,alpha=0.3)
        ax0.set_yticklabels([])
        ax0.set_xticklabels([])
        ax0.set_zticklabels([])
        ax0.text2D(0.05, 0.87, kwargs.get('ax0label',None),transform=ax0.transAxes)
        
        # Evaluate function at points and make plot
        EVAL = func(X,Y,Z, **kwargs)
        # EVAL can be of various types, handled below
        Emin =[]; Emax=[]
        if isinstance(EVAL,list):
            for p in EVAL:
                if isinstance(p,dict):
                    whichax = p.get('axis',1)
                    axp = ax2 if whichax ==2 else ax1
                    porder = p.get('zorder',2)
                    labelstr = p.get('label',None)
                    
                    fill = p.get('fill', False)
                    if fill:
                        ydat = p.get('y',None)
                        if ydat is not None:
                            axp.plot(t,ydat[0],
                                     lw=p.get('lw',2.),\
                                     color=p.get('color','black'),\
                                     alpha=p.get('fillalpha',0.5),\
                                     zorder=porder,\
                                     label=labelstr
                                     )
                            axp.fill_between( t, ydat[0], ydat[1],\
                                              lw=p.get('lw',2.),\
                                              color=p.get('color','black'),\
                                              facecolor=p.get('fillcolor','gray'),\
                                              alpha=p.get('fillalpha',0.5),\
                                              zorder=porder
                                            ) 
                            Emin.append( min( ydat[0].min(), ydat[1].min() ))
                            Emax.append( max( ydat[0].max(), ydat[1].max() )) 
                    else:
                        ydat = p.get('y',None)
                        if ydat is not None:
                            axp.plot( t, ydat,\
                                      lw=p.get('lw',2.),\
                                      color=p.get('color','black'),\
                                      alpha=p.get('alpha',1.0),\
                                      zorder=porder,\
                                      label=labelstr
                                    )
                            Emin.append( ydat.min() ) 
                            Emax.append( ydat.max() ) 
                      
                else:
                    ax1.plot(t,p); Emin.append(p.min()); Emax.append(p.max())
            
        elif len(EVAL.shape) > 1 :
            for i,E in enumerate(EVAL):
                ax1.plot(t,E); Emin.append(E.min()); Emax.append(E.max())
                
        else:
            ax1.plot( t, EVAL); Emin.append(E.min()), Emax.append(E.max())
            
        Emin = min(Emin); Emax=max(Emax)
        dE = Emax-Emin
        
        ax1.grid()
        ax1.set_xlabel('$\mu\mathrm{m}$', fontsize=14)
        ax1.set_ylabel( self.unitlabel, rotation=0, fontsize=14, labelpad=-5 )
        
        
        # Finalize figure
        if gs is None:
            gs.tight_layout(fig, rect=[0.,0.,1.0,1.0])
            return Emin-0.05*dE, Emax+0.05*dE, ax1
        else:
            ax2.xaxis.set_major_locator( matplotlib.ticker.MultipleLocator(20) ) 
            ax2.xaxis.set_minor_locator( matplotlib.ticker.MultipleLocator(10) ) 
            return Emin-0.05*dE, Emax+0.05*dE, ax1, ax2
        
    def plot3Line(self,**kwargs):
        fig = plt.figure(figsize=(12,8.))
        gs = matplotlib.gridspec.GridSpec( 2,2, wspace=0.2)
        
        direcs = [ (np.pi/2, 0.), (np.pi/2, np.pi/2), (0., np.pi), (np.arctan(np.sqrt(2)), np.pi/4) ]
        ax0labels = ['$\mathbf{100}$', '$\mathbf{010}$', '$\mathbf{001}$', '$\mathbf{111}$' ]
                     
        gsindex = [ (0,0), (1,0), (0,1), (1,1) ] 
        
        Ax1 = []; Ax2 = []
        Ymin =[]; Ymax=[]
        for i in range(4):
            ymin, ymax, ax1, ax2 = self.plotLine( direc=direcs[i], gs= gs[gsindex[i]], figGS=fig, ax0label=ax0labels[i], **kwargs) 
            Ymin.append(ymin); Ymax.append(ymax); Ax1.append(ax1); Ax2.append(ax2)

        Ymin = min(Ymin); Ymax = max(Ymax)
        for ax in Ax1:
            ax.set_ylim( Ymin, Ymax)
            
        Ax1[0].legend( bbox_to_anchor=(1.06,0.005), \
            loc='lower right', numpoints=1, labelspacing=0.2,\
             prop={'size':10}, handlelength=1.1, handletextpad=0.5 )
        Ax2[0].legend( bbox_to_anchor=(1.10,1.10), \
            loc='upper right', numpoints=1, labelspacing=0.2, \
             prop={'size':10}, handlelength=1.1, handletextpad=0.5 )
            
        gs.tight_layout(fig, rect=[0.,0.,1.0,1.0])
         
    
        
    def plotCross(self,
            axes = None,
            func = None,
            origin  = vec3(0.,0.,0.), 
            normal   = (np.pi/2, 0),
            lims0   = (-50,50), 
            lims1   = (-50,50),
            extents = None,
            npoints = 240 ):
        
        if func is None:
            func = self.evalpotential
        
        if extents is not None:
            lims0 = (-extents, extents)
            lims1 = (-extents, extents)
        
        # Make the unit vectors that define the plane
        unit = vec3()
        th = normal[0]
        ph = normal[1]
        unit.set_spherical( 1, th, ph) 
        #orth1 = -1.*vec3( np.cos(th)*np.cos(ph) , np.cos(th)*np.sin(ph), -1.*np.sin(th) )
        orth0 = vec3( -1.*np.sin(ph), np.cos(ph), 0. )
        orth1 = cross(unit,orth0)
        
        t0 = np.linspace( lims0[0], lims0[1], npoints )
        t1 = np.linspace( lims1[0], lims1[1], npoints ) 
        
        # Obtain points on which function will be evaluated
        T0,T1 = np.meshgrid(t0,t1)
        X = origin[0] + T0*orth0[0] + T1*orth1[0] 
        Y = origin[1] + T0*orth0[1] + T1*orth1[1]
        Z = origin[2] + T0*orth0[2] + T1*orth1[2] 
        
    
        if axes is None:
            # Prepare figure
            fig = plt.figure(figsize=(9,4.5))
            gs = matplotlib.gridspec.GridSpec( 1,2, wspace=0.2)
        
            ax0 = fig.add_subplot( gs[0,0], projection='3d')
            ax1 = fig.add_subplot( gs[0,1])
        else:
            ax0 = axes[0]
            ax1 = axes[1]
        
        
        # Plot the reference surface
        ax0.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3, linewidth=0.)
        ax0.set_xlabel('X')
        ax0.set_ylabel('Y')
        ax0.set_zlabel('Z')
        lmin = min([ ax0.get_xlim()[0], ax0.get_ylim()[0], ax0.get_zlim()[0] ] )
        lmax = max([ ax0.get_xlim()[1], ax0.get_ylim()[1], ax0.get_zlim()[1] ] )
        ax0.set_xlim( lmin, lmax )
        ax0.set_ylim( lmin, lmax )
        ax0.set_zlim( lmin, lmax )
        ax0.set_yticklabels([])
        ax0.set_xticklabels([])
        ax0.set_zticklabels([])
        
        
        # Evaluate function at points and plot
        EVAL = func(X,Y,Z)
  
        T1_1d = np.ravel(T1)
        EVAL_1d = np.ravel(EVAL)
        im =ax1.pcolormesh(T0, T1, EVAL, cmap = plt.get_cmap('rainbow'))
        plt.axes( ax1)
        plt.colorbar(im)
        
        if axes is None:
            # Finalize figure
            gs.tight_layout(fig, rect=[0.,0.,1.0,1.0])
            
        return EVAL.min(), EVAL.max(), im

    def plot3Cross(self, **kwargs):
        fig = plt.figure(figsize=(8,8))
        gs = matplotlib.gridspec.GridSpec( 3,2, wspace=0.2)
        
        normals = [ (np.pi/2, 0.), (np.pi/2, -np.pi/2), (0., -1.*np.pi/2) ] 
        yMin = 1e16
        yMax =-1e16
        Ims = []
        for i in range(3):
            ax0 = fig.add_subplot( gs[i,0], projection='3d')
            ax1 = fig.add_subplot( gs[i,1]); 
            ymin, ymax, im = self.plotCross( normal=normals[i], axes=(ax0,ax1), **kwargs) 
            Ims.append(im)
            if ymin < yMin: yMin = ymin
            if ymax > yMax: yMax = ymax
        #for im in Ims:
        #    im.set_clim( vmin=yMin, vmax=yMax)
        gs.tight_layout(fig, rect=[0.,0.,1.0,1.0])

# <codecell>

class simpleCubic( potential ):
    def __init__(self, **kwargs):
        """Simple cubic lattice potential """
        
        # Initialize lattice part 
        axes= [ (np.pi/2,0.), (np.pi/2, np.pi/2), (0,0) ] 
        self.l  = kwargs.get('wavelength', 1.064)
        self.m  = kwargs.get('mass', 6.)
        self.w  = kwargs.get('waists', ((40.,40.), (40.,40.), (40.,40.)) )
        self.Er = kwargs.get('Er', (7.,7.,7.) )
        self.r  = kwargs.get('retro', (1.,1.,1.) )
        self.a  = kwargs.get('alpha', (1.,1.,1.) )
        self.scale = kwargs.get('scale', 10.) 
        
        self.Er0 = Erecoil(self.l, self.m)
        
        lattbeams = [ LatticeBeam( axis=axes[i], Er=self.Er[i], wavelength=self.l, scale=self.scale,\
                                   waists=self.w[i], retro=self.r[i], alpha=self.a[i] ) for i in range(3) ] 
        
        potential.__init__(self, lattbeams, units=('$E_{R}$', 1/self.Er0) )
        
        self.GRw  = kwargs.get('green_waists', ((36.,36.), (36.,36.), (36.,36.)) ) 
        self.GREr = kwargs.get('green_Er', (4.0, 4.0, 4.0) ) 
        self.GRl  = kwargs.get('green_wavelength', 0.532)
        
        # Express the power requiered for each GR beam, given the Er compensation value
        # in units of the lattice recoil, and given the GR beam waists
        GRmW = [ 1000.* self.GREr[i]*  self.Er0/np.abs(U2uK(self.GRl)*2/np.pi) \
                      * self.GRw[i][0]*self.GRw[i][1]  for i in range(3)]
        
        self.greenbeams = [  GaussBeam( axis=axes[i], mW=GRmW[i], waists=self.GRw[i], wavelength=self.GRl) for i in range(3) ]
        
    def Bottom0( self ):
        EVAL = 0.
        for b in self.beams:
            EVAL += b.getBottom( 0., 0., 0.)
        for g in self.greenbeams:
            EVAL += g(0.,0.,0.)
        EVAL = EVAL*self.unitfactor
        return EVAL
            
        
    def Bottom( self, X, Y, Z):
        EVAL = np.zeros_like(X)
        for b in self.beams:
            EVAL += b.getBottom( X, Y, Z)
        for g in self.greenbeams:
            EVAL += g(X,Y,Z)
        return EVAL*self.unitfactor
    
    def V0( self, X, Y, Z):
        EVAL = []
        for b in self.beams:
            EVAL.append( b.getV0( X, Y, Z)*self.unitfactor )
        return np.array(EVAL)
            
    
    def LatticeMod( self, X, Y, Z):
        V0s = self.V0( X, Y, Z )
        Mod = np.amin(V0s, axis=0)
        return self.Bottom(X,Y,Z) + Mod*np.power( np.cos( 2.*np.pi*np.sqrt(X**2 + Y**2 + Z**2 ) / self.l / self.scale ), 2)
    
    def Bands( self, X, Y, Z, **kwargs):
        NBand = kwargs.get('NBand',0.)
        
        self.V0_ = self.V0(X,Y,Z) 
        self.Bottom_ = self.Bottom(X,Y,Z)
        self.LatticeMod_ = self.Bottom_ + np.amin(self.V0_,axis=0)*\
                           np.power( np.cos( 2.*np.pi*np.sqrt(X**2 + Y**2 + Z**2 ) / self.l / self.scale ), 2)
        
        bands = bands3dvec( self.V0_, NBand=0 ) 
        
        # The zero of energy for this problem is exactly at the center
        # of the lowest band.  This comes from the t_{ii} matrix element
        # which is generally neglected in the Hamiltonian. 
        self.Ezero = (bands[1]+bands[0])/2. + self.Bottom_
        self.Ezero0 = self.Ezero.min()

        
        # globalMu is given in Er, and is measured from the value
        # of Ezero at the center of the potential
        # When using it in the phase diagram it has to be changed to
        # units of the tunneling
        self.globalMu = kwargs.get('globalMu', 0.15) 
        self.globalMu = self.Ezero0 + self.globalMu
        
        # The threshold for evaporation can ge obtaind from 
        # the bottom of the band going far along one beam
        self.evapTH = bands3dvec( self.V0( 100., 0., 0. ), NBand=0 )[0] + self.Bottom(100.,0.,0.)
        
        
        
        
        tunneling = (bands[1]-bands[0])/12. 
        onsite = Onsite( self.V0_, a_s=300., lattice_spacing= self.l/2 )
        localMu = self.globalMu - self.Ezero 
        
        onsite_t = onsite / tunneling
        localMu_t = localMu / tunneling
        
        density  = getHTSEInterp( 1.6, name="density" )( np.column_stack(( np.ravel(onsite_t),np.ravel(localMu_t) )) ).reshape(onsite_t.shape)
        doublons = getHTSEInterp( 1.6, name="doublons")( np.column_stack(( np.ravel(onsite_t),np.ravel(localMu_t) )) ).reshape(onsite_t.shape)
        entropy  = getHTSEInterp( 1.6, name="entropy" )( np.column_stack(( np.ravel(onsite_t),np.ravel(localMu_t) )) ).reshape(onsite_t.shape)
        
        # Higher zorder puts stuff in front
        return [ 
                 {'y':(bands[0] + self.Bottom_, self.Ezero ), 'color':'blue', 'lw':2., \
                  'fill':True, 'fillcolor':'blue', 'fillalpha':0.75,'zorder':10, 'label':'$\mathrm{band\ lower\ half}$'},
                 
                 {'y':(self.Ezero + onsite, bands[1] + self.Bottom_+onsite), 'color':'purple', 'lw':2., \
                  'fill':True, 'fillcolor':'plum', 'fillalpha':0.75,'zorder':10, 'label':'$\mathrm{band\ upper\ half}+U$'},
                 
                 #{'y':localMu, 'color':'green', 'lw':1.0, 'zorder':1.9} ,\
                 {'y':np.ones_like(X)*self.globalMu, 'color':'limegreen','lw':2,'zorder':1.9, 'label':'$\mu_{0}$'},
                 {'y':np.ones_like(X)*self.evapTH, 'color':'#FF6F00', 'lw':2,'zorder':1.9, 'label':'$\mathrm{evap\ th.}$'},
                 
                 {'y':self.Bottom_,'color':'gray', 'lw':0.5,'alpha':0.5},
                 {'y':self.LatticeMod_,'color':'gray', 'lw':1.5,'alpha':0.5,'label':r'$\mathrm{lattice\ potential\ \ }\lambda\times10$'},
                 
                 #{'y':onsite, 'color':'red', 'lw':3.0},\
                 #{'y':onsite/(12*tunneling), 'color':'black', 'lw':1.5, 'axis':2},
                 
                 {'y':density,  'color':'blue', 'lw':1.75, 'axis':2, 'label':'$n$'},
                 {'y':doublons, 'color':'red', 'lw':1.75, 'axis':2, 'label':'$d$'},
                 {'y':entropy,  'color':'black', 'lw':1.75, 'axis':2, 'label':'$s$'}  
                  ]
    
    
# TO DO NEXT
# 1. put the actual parameters for our experiment (start with what I have in mathematica)
# 2. find a way to only plot the (111)
# 2a. put big labels of the main parameters (lattice depth, a0, waists? )
# 3. Make good plots of density and the other thermodynamic quantities
# 4. Integrate the thermodynamic quantities over the volume
# 5. Get column density profiles of density and doublons






            
    
latt3d = simpleCubic( )
#latt3d.plot3Line()
#latt3d.plotCross( normal=(0.,0.), lims0=(-50,50), lims1=(-50,50), npoints=240)
#latt3d.plot3Line(func=latt3d.Bottom)
#latt3d.plot3Line(func=latt3d.LatticeMod)
latt3d.plot3Line(func=latt3d.Bands, globalMu=0.23)

# <codecell>

matplotlib.gridspec.GridSpecFromSubplotSpec??

# <codecell>

# Here the interpolation data for the band structure is loaded from disk
v0 = np.loadtxt('banddat/interpdat_B3D_v0.dat')
NBands = 3
from scipy.interpolate import interp1d
interp0 = []
interp1 = []
for n in range( NBands ):
    interp0.append( interp1d(v0, np.loadtxt('banddat/interpdat_B3D_0_%d.dat'%n) ))
    interp1.append( interp1d(v0, np.loadtxt('banddat/interpdat_B3D_1_%d.dat'%n) ))
    
# Using the interolation data calculate a function that will get the bottom and top
# of the 3D band in a vectorized way. 

def bands3dvec( v0, NBand=0 ):
    assert len(v0)==3
    bandbot = np.zeros_like( v0[0] ) 
    bandtop = np.zeros_like( v0[0] ) 
    if NBand == 0:
        nband = [0, 0, 0]
    elif NBand == 1:
        v0.sort(axis=0)
        nband = [1, 0, 0]
    else:
        return None
    for i in range(3):
        in1d = nband[i] 
        if in1d%2 ==0:
            bandbot += interp0[in1d](v0[i]) 
            bandtop += interp1[in1d](v0[i])
        else:
            bandbot += interp1[in1d](v0[i])
            bandtop += interp0[in1d](v0[i])
    return np.array((bandbot,bandtop))

# <codecell>

#Here the interpolation data for the on-site interactions is loaded from disk
from scipy.interpolate import interp1d
wFInterp = interp1d( np.loadtxt('banddat/interpdat_wF_v0.dat'), np.loadtxt('banddat/interpdat_wF_wF.dat'))

# Using the interpolation data calculate a function that will get the on-site
# interactions in a vectorized way. 

def Onsite( v0,  a_s=300., lattice_spacing=0.532):
    assert len(v0)==3
    wint = np.ones_like( v0[0] ) 
    for i in range(3):
        wint *= wFInterp( v0[i] )
    # The lattice spacing is given in um
    a0a = 5.29e-11 / (lattice_spacing*1e-6)
    return a_s * a0a * np.power(wint, 1./3.)

# <codecell>

localmu = np.array( [[ 0.0, 1.],[2.,3.]] )
onsite = np.array( [[ 10.,10.],[10.,10.]] )
print localmu
print onsite

# <codecell>

getHTSEInterp( Temperature, name="density")( np.column_stack(( np.ravel(onsite),np.ravel(localmu) )) ).reshape(onsite.shape)

# <codecell>

Temperature = 2.4
fdens = getFuchsInterp( 2.4 , name='density')
fHdens = getHTSEInterp( 2.4 , name='density')
print fHdens(10.,0.)
print fHdens(10.,1.)
print fHdens(10.,2.)
print fHdens(10.,3.)

# <codecell>

# Loading up the phase diagram interpolation functions and some examples of how to use them

from HubbardPhaseDiagramInterp import getFuchsInterp, getHTSEInterp

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

# <codecell>

l = [ {'a':0,'b':1}, {'c':2,'d':3} ] 

# <codecell>

type(l)

# <codecell>

print type(l) == type([])

# <codecell>


