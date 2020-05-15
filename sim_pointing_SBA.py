# C. Hales, NRAO
# version 1.0 (24 April 2020)
#
# Measure image dynamic range and fidelity achievable with different dynamic pointing
# errors for ngVLA SBA revC (19 x 6m).  The procedure involves constructing mis-pointed
# antenna voltage patterns, then corrupted baseline primary beams.  For each baseline,
# the corrupted baseline power pattern is multipled with the true sky and used to predict
# a corrupted visibility.  Both amplitude and phase errors arising from pointing errors
# are included.
#
# This code is released with ngVLA Memo 75
#   https://library.nrao.edu/public/memos/ngvla/NGVLA_75.pdf
# and is available at:
#   https://github.com/chrishales/ngVLA-pnt-err-img-sims
#
# related: * CASR-471
#          * EVLA Memo 84
#          * https://ieeexplore.ieee.org/document/7762834
#          * ngVLA Memo 60
#          * ngVLA Memo 67
#
# requires: Python 3.6 and CASA 6.0
# suggested: build the tools and tasks for CASA 6.0 from modified source (see reason below)
# https://casa.nrao.edu/casadocs/casa-5.6.0/introduction/casa6-installation-and-usage
# https://gitlab.nrao.edu/rurvashi/simulation-in-casa-6/blob/master/Simulation_Script_Demo.ipynb
#
# simulation details:
# - only pointing errors; no noise or other errors (ie no bandwidth or time smearing,
#      no tropospheric amplitude or phase fluctuations, etc)
# - random pointing error assigned per antenna, per coherence timescale.  This code
#      assumes statistically independent pointing errors per integration (ie the pointing
#      error coherence timescale is <= integration time).  Pointing errors are injected
#      in the X-Y linear-pol antenna frame, subject to parallactic angle rotation, so
#      that statistics are correct over a long observation (this code will assume that
#      the X feed is aligned along meridian at zero parallactic angle, though it doesn't
#      really matter which is which for this simulation)
# - Stokes I only
# - observation duration is centered about zero hour angle (when observing the center of
#      the mosaic; avoids shadowing) with transit at 10 deg from zenith
# - performed at 27 GHz (ngVLA band~4)
# - two spectral arrangements:
#      (i)  single narrow-band channel
#      (ii) wide fractional bandwidth (13/27 = 48%) with power law emission
#           (band 4: 20.5 - 34 GHz, max instantaneous bw 13.5 GHz; this sim samples 13 GHz)
# - two pointing arrangements (hardcoded) with pointing separation HPBW(@27GHz)/sqrt(2):
#      (i)  7 pointings in hexagonal pattern, source centered in the central pointing
#      (ii) 3 pointings in triangle, source centered in middle of them
# - each pointing is observed for 1 minute with zero slew time, cycling through all
#   pointings and repeating until the observation ends
# - idealized extended source comprising a sum of 3 point sources located in a
#      linear arrangement with maximum angular extent between injected source
#      locations equal to 50% of the LAS(@27GHz) of the SBA.  For reference:
#                   20 GHz: PSF ~= 0.9' , LAS ~= 4.7' , HPBW ~= 8.6'
#                   27 GHz: PSF ~= 0.6' , LAS ~= 3.5' , HPBW ~= 6.4'
#                   34 GHz: PSF ~= 0.5' , LAS ~= 2.8' , HPBW ~= 5.1'
# - all antennas are assumed to have identical apertures that are circular, unblocked,
#      and with zero squint.
#      warning: Currently, tclean will override the unblocked aperture that this code
#               defines using the vpmanager and instead assume a blocked aperture.
#               The logger warning states:
#                  INFO task_tclean::HetArrConvFunc::findAntennaSizes  Overriding PB
#                  with Airy of diam,blockage=6 m,0.375 m starting with antenna 0
#               This originates in code/synthesis/TransformMachines/HetArrayConvFunc.cc
#               where the blockage is forcefully set to the ALMA ratio of blockage to
#               dish (antdia/12*0.75) whenever the code encounters an AIRY pbClass.
#               I have requested a fix in CAS-11464.  In the meantime, to overcome
#               this, you need to build a modified tclean from source, which requires
#               installing the casa 6 tools and tasks from source.  Oh joy!
#            https://open-bitbucket.nrao.edu/projects/CASA/repos/casa6/browse/casatools
#            https://open-bitbucket.nrao.edu/projects/CASA/repos/casa6/browse/casatasks
#               Prior to building anything, modify HetArrayConvFunc.cc to avoid forcing
#               the ALMA blockage ratio.  Search for code "dishDiam(k)/12.0*.75" .
#               The quickest approach is to change 0.75 --> 0 in the multiplication.
#               You will still see tclean override warnings in the logger, but at least
#               the correct voltage patterns will be used.
# - rather than FT phase-corrupted aperture illumination functions per antenna to produce
#      mis-pointed antenna primary beams (the relevant fully-functional code for which is
#      commented out below, in case you are curious; search for the word "legacy"), this
#      code simply offsets airy disks to produce the latter.  This allows for much greater
#      angular resolution in the primary beams and in turn fewer computational artifacts
#      that would otherwise bias the dynamic range and image fidelity measurements (see
#      additional commentary above function makeVolt).  This is possible because this
#      code assumes unblocked circular apertures with zero squint.
# - the casa logfile produced by this code only contains output from imaging.  All other
#      processing details are suppressed.  You will see some imaging warnings printed to
#      screen; these are benign, unless you see failures being reported.
# - the observatory center will be defined as the mean of the antenna positions, not the
#      ngVLA location pre-defined in CASA.
#      fyi: listobs, plotms etc will use the CASA internally-defined location of the
#           telescope name listed in the header of the config file.  You can overwrite the
#           telescope name in this code (see input below), but then some tasks like listobs
#           will segfault (yay).  This code will manually set the center of the array to
#           the mean antenna position (in this case for the SBA), which is different to the
#           ngVLA location stored in CASA.  Keep all this in mind if you notice curious
#           "features" eg when plotting things using plotms (I've documented a case
#           involving hour angle inline below).  Enjoy!
# - imaging uses tclean with the 'mosaic' gridder (no widefield A or W effects) and
#      natural weighting.  To significantly improve runtime this code uses the im.ft
#      tool rather than tclean for the visibility prediction step.  The former assumes
#      gridder='standard', which isn't the same as 'mosaic' in the imaging step.
#      However, this doesn't impose any dynamic range or fidelity penalties.  See
#      function simVis() for more details.
# - clean mask applied in tclean to prevent cleaning sidelobes, given by circle about
#      injected source locations with radius given by the larger of 2*cell_pb and
#      perr/sqrt(Nant).  The default threshold assigned below is the smallest value
#      appropriate before sidelobes will begin to be cleaned.  The numerical noise in
#      an image residual resulting from a single channel zero pointing error simulation
#      is just below 1e-3, yielding a dynamic range limit of 30 dB.  Consult ngVLA Memo
#      60 to ensure that you aren't setting parameters that will result in unachievable
#      dynamic range parameter space.
# - parallel processing using python multiprocessing, will use num_cores-2 leaving 2
#      free so your OS doesn't freeze.  Note that imaging is not parallelized here
#      (but not a major time sink; casampi is not used here).  As a guide, this code
#      can process 1 pointing of a 3-pointing mosaic in 1 channel and 1 timestep with
#      4096x4096 arrays running in parallel with 38 cores in 1 min (this doesn't include
#      overheads from code startup or later imaging; these aren't significant anyway).
#      This is read/write limited rather than CPU-limited.  The runtime above rises
#      to around 3.5 mins per channel when processing a wideband 13-channel simulation.
# - memory management has been considered, but it could be better.  I ran this code
#      on a machine with 256 GB RAM.  Anything less will probably swap.  I reduced
#      imsize_pb to 2048 and set cell_pb = 0.8/3600 for the wideband 13-channel sims.
# - dynamic range is defined in calcStats as image_max / residual_rms , where the image
#      and residual are flatnoise tclean outputs (ie not PB corrected).  For wideband
#      sims, residual.tt0 is not the same as the residuals in image.tt0 because the
#      latter contains the combination from higher order residuals.  To compensate,
#      the wideband results obtained in this code estimate the rms in image.tt0 by
#      measuring the median absolute deviation of all pixels less than 0.1 Jy/beam
#      and multiplying this by factor 1.48.  This factor will recover the standard
#      deviation for ideally normally distributed data.  The standard deviation is
#      not the same as rms, but should be approximately the same for this simulation
#      with expected zero mean & median.  (The rms isn't extracted because the few
#      pixels near the 0.1 cutoff will otherwise bias the results.)
# - fidelity is defined in calcStats following the F_3 metric from ngVLA Memo 67,
#      but with some corrections.  See Equation 7 in that memo for the notation.
#      Here, image M is the true sky model attenuated by the PB and only then
#      convolved with the PSF.  And here, to be consistent, image I is the
#      PB-uncorrected dirty image.  A nice thing about these definitions is that
#      this image fidelity metric is insensitive to deconvolution errors.
#      The sums over pixels are only performed within the pbthresh=50% mosaic
#      PB contour, following ngVLA Memo 67.  The 70% contour comfortably encloses
#      the injected sources in this simulation for the 3 and 7 pt mosaics.

#########################################################################################

antloc  = 'ngvla-sba-revC.cfg'  # include full path if not in cwd
                                # Note: this simulation will not assign the
                                #       observatory location for NGVLA
                                #       defined inside CASA, but will instead
                                #       assign it as the mean antenna position

telname = 'NGVLA_SBA'    # rename header from antloc
                         #    if this includes '*NGVLA*' then some tasks like plotms
                         #    and listobs will do their own thing when performing
                         #    coordinate calculations.  Accept for now, but beware!
                         #    FYI this code is internally consistent regardless
                         #    of telname.  Note that tclean is also affected by this,
                         #    but not for the continuum imaging of interest here.

perr = [0,10,20,40,60]   # list of pointing errors, radial rms in arcsec            ################## [0,10,20,40,60]
freq      = 27           # central frequency of band, in GHz
spwname   = 'Band4'      # spectral window name to record in MS
chanbw    = 1000         # channel bandwidth, in MHz
                         #   (only needed for multi-channel spacing)
nchan     = [1]          # list containing number of channels across band
                         #   (odd numbers only, eg [1,13])
pointings = [3]          # list containing number of pointings to simulate
                         #   (must be [3], [7], or [3,7]; these values reflect
                         #    structure hardcoded in pntCntrs)
obsdur    = 3            # total observation duration, in minutes
                         #   (obsdur/x must return an integer where x is given
                         #    by the integer(s) specified in pointings)
tint      = 60           # integration time, in seconds
                         #   (60/tint must return an integer because the
                         #    observation time per pointing is hardcoded
                         #    to be 1 minute)
spindx    = -0.8         # spectral index S \propto nu**spindx for injected sources

shadowlimit    = 0.01    # antenna shadow limit
elevationlimit = '45deg' # antenna elevation limit, visibilities flagged below this

# size for primary beams (legacy: also used for aperture illumination functions)
# to make life easier, the same coordinate frame will be used to sample
# the injected model (see makeComponentList for details).
cell_pb   = 0.4/3600     # deg; ensure at least 100 pixels across HPBW and
                         #    at least 3 pixels per PSF FWHM.
                         #    more strict: ensure a few pixels per perr
                         #   (airy disk HPBW ~= 8.6'/6.4'/5.1')
imsize_pb = 4096         # ensure that cell_pb*imsize_pb is around 3 x HPBW so that
                         #    all pointings are included in the FOV
                         #   (even numbers only)
                         #    legacy: make cell_pb*imsize_pb as large as possible,
                         #    at least ~20x HPBW, so that the aperture illumination
                         #    functions are sampled by at least a few dozen pixels
                         #   (don't go larger than 4096, or even 2048, or else you
                         #    will probably run out of memory)
cleanfrac = [0.002,0.002,0.006,0.01,0.02]                         ################################### [0.002,0.002,0.006,0.01,0.02]
                         # cleaning threshold as fraction of peak in dirty image,
                         #   corresponding to each perr above (independent of
                         #   nchan or pointings).  Ensure cleanfrac doesn't exceed
                         #   the dynamic range of the simulation (will require
                         #   some experimentation, set imgOnly=True to rerun
                         #   simulation with imaging only).  Note that tclean
                         #   auto-multithresh is not used here.
                         #   (DR limit of this code ~= 0.001 for perr=0)
                         #   I used the following:
                         #   (pnts=pointings, nint=obsdur/tint)
                         #   * perr=[0,10,20,40,60], pnts=[7], nint=7, nchan=[1]
                         #     cleanfrac=[0.002,0.004,0.01,0.02,0.04]
                         #   * perr=[0,10,20,40,60], pnts=[3], nint=3, nchan=[1]
                         #     cleanfrac=[0.002,0.002,0.006,0.01,0.02]
                         #   * perr=[0,10,20,40,60], pnts=[3], nint=30, nchan=[1]
                         #     cleanfrac=[0.002,0.002,0.002,0.005,0.007]
                         #   * perr=[0,10,20,40,60], pnts=[3], nint=3, nchan=[13],
                         #     cleanfrac=[0.002,0.005,0.008,0.05,0.05]
pbthresh  = 0.8          # mosaic PB contour in which to calculate fidelity stats         ########### 0.5
nterms    = 3            # number of taylor terms for cleaning multi-channel cases
                         #   (select >2 to ensure expansion errors don't dominate)

testing   = False        # write out fits images for the model, unperturbed
                         #   (template) ant pb (legacy: and AIF) and, for a nominated
                         #    time step only (testingT), the ant + bl pb's
                         #   (True/False)
testingT  = 1            # integration number (0-based) to write images above,
                         #    noting that parang~=0 (depending on pointing center)
                         #    is at testingT = int((obsdur*60/tint-1)/2)
rapidtest = False        # if True, and if testing=True, ignore obsdur and only
                         #    run the simulation at a single timestep (and single
                         #    pointing; consider cleanfrac) at timestep = testingT
imgOnly   = True        # If True, don't run simulations, only perform imaging and         ############ False
                         #    write out new results.  If False, include simulations.
                         #    You must run with False prior to running with True.

#########################################################################################

import os,pickle,re
import numpy as np
#legacy: from scipy import fftpack
from scipy.special import j1
import multiprocessing,ctypes
from functools import partial
from astropy import units as u
from astropy.coordinates import Angle,ITRS,SkyCoord
from casatools import componentlist,image,imager,table,measures
from casatools import ms,quanta,simulator,vpmanager
from casatasks import importfits,exportfits,split,casalog,tclean
from casatasks import flagdata,imstat,immath,imhead,widebandpbcor
from casatasks.private import simutil

# parallel processing, retain 2 cores to avoid freezing other OS tasks
cpuCount   = os.cpu_count()-2

prefix     = 'extended-source'
c0         = 299792458   # m/s
npol       = 1           # 2 polarizations (X,Y) but both are
                         # equally affected by ant pointing error.
                         # This code will therefore apply the same
                         # AIF/PB to each polarization.  Retain plumbing
                         # in case npol differentiation needed in future

#aifdir = prefix+'_ant-aif'   # legacy
pbAdir = prefix+'_pb-ant'
pbBdir = prefix+'_pb-bl'
if testing:
    # legacy:
    ## antenna aperture illumination functions
    ## ignore the dummy coordinates that will come with these fits images
    #os.system('rm -rf '+aifdir)
    #os.system('mkdir '+aifdir)
    
    # antenna primary beams
    os.system('rm -rf '+pbAdir)
    os.system('mkdir '+pbAdir)
    
    # baseline primary beams
    os.system('rm -rf '+pbBdir)
    os.system('mkdir '+pbBdir)

cl = componentlist()
ia = image()
im = imager()
me = measures()
ms = ms()
qa = quanta()
tb = table()
sl = simutil.simutil()
sm = simulator()
vp = vpmanager()

# set RA and Dec at center of central pointing
# assign former at transit and latter using mean ITRF-phi of SBA offset by -10 deg
# note ITRF gives spherical coordiantes, so phi is only latitude on spherical earth
# Don't use EarthLocation.from_geocentric which returns lat lon using oblate WGS84.
# note that I could alternatively obtain the ngVLA origin using (ITRF-based)
# me.observatory('ngVLA') but this won't return the mean SBA phi
# If you use the following it will assign the obscoord that is pre-defined
# inside CASA for the telname in the header (ie NGVLA)
#(x,y,z,d,antname,telname,obscoord) = sl.readantenna(antloc)
# Instead, take the antenna coordinates and assign our own observatory location
x          = np.loadtxt(antloc,usecols=0)
y          = np.loadtxt(antloc,usecols=1)
z          = np.loadtxt(antloc,usecols=2)
d          = np.loadtxt(antloc,usecols=3,dtype=int)
antname    = np.loadtxt(antloc,usecols=4,dtype=str)
# assign observatory location at mean antenna position
obscoord   = me.position('ITRF',str(np.mean(x))+'m',
                                str(np.mean(y))+'m',
                                str(np.mean(z))+'m')
nant       = len(antname)
antdia     = d[0]                     # all same, in meters
phi_mean   = np.rad2deg(ITRS(x,y,z).spherical.mean().lat.value)
ra_transit = Angle(0,u.hourangle)     # center sources at RA = 0
                                      #  (observe HA = -obsdur/2 --> +obsdur/2)
decl       = Angle(phi_mean-10,u.deg) # keep dec constant for all injected gaussians
c_transit  = SkyCoord(ra_transit,decl,frame='icrs')
ntsteps    = np.int(obsdur*60/tint)   # total number of time steps
minsep     = 1e10
for a1 in range(nant):
    for a2 in range(nant):
        if a1 < a2:
            sep = np.sqrt((x[a1]-x[a2])**2+(y[a1]-y[a2])**2+(z[a1]-z[a2])**2)
            if sep < minsep: minsep = sep

las0  = np.rad2deg(c0/(freq*1e9)/minsep)  # deg
hpbw0 = np.rad2deg(c0/(freq*1e9)/antdia)  # deg

# sky-frame coordinates U and V for PBs, where +u = +E and +v = +N, in radians
fov         = np.deg2rad(cell_pb)*imsize_pb
# arrays to/from fits are rotated by 90 deg compared to the image frame, so
# counteract by saving to uu,vv instead of expected vv,uu (the latter which
# would require np.rot90 to be called later)
# pixel (0,0) in lower left corner
uu,vv       = np.mgrid[:imsize_pb,:imsize_pb]
uvals       = (np.arange(-fov/2,fov/2,np.deg2rad(cell_pb))+np.deg2rad(cell_pb)/2)*-1
vvals       =  np.arange(-fov/2,fov/2,np.deg2rad(cell_pb))+np.deg2rad(cell_pb)/2
uvals0_base = multiprocessing.Array(ctypes.c_double,imsize_pb**2)
uvals0      = np.ctypeslib.as_array(uvals0_base.get_obj())
uvals0      = uvals0.reshape(imsize_pb,imsize_pb)
uvals0      = uvals[uu]
vvals0_base = multiprocessing.Array(ctypes.c_double,imsize_pb**2)
vvals0      = np.ctypeslib.as_array(vvals0_base.get_obj())
vvals0      = vvals0.reshape(imsize_pb,imsize_pb)
vvals0      = vvals[vv]

# legacy:
# sky-frame coordinates U and V for AIFs
#cell_uv = 1/np.deg2rad(fov)          # wavelengths
#uvmax   = cell_uv*imsize_pb/2        # wavelengths [= 1/np.deg2rad(cell_pb)/2]
#vv,uu   = np.mgrid[:imsize_pb,:imsize_pb]
#uvals   =  np.arange(-uvmax, uvmax, cell_uv) + cell_uv/2        # +u = +W
#vvals   = (np.arange(-uvmax, uvmax, cell_uv) + cell_uv/2)*-1    # +v = +N
#uvals0  = uvals[uu]
#vvals0  = vvals[vv]
#radius2 = (uu-(imsize_pb-1)/2)**2 + (vv-(imsize_pb-1)/2)**2

# legacy: don't need this to be a complex array now that complex voltage patterns
#         are no longer returned from the FT of AIFs
#volt_base  = multiprocessing.Array(ctypes.c_double,imsize_pb**2*nant*npol*2)
#volt       = np.ctypeslib.as_array(volt_base.get_obj())
#volt       = volt.view(np.complex128).reshape(imsize_pb,imsize_pb,nant,npol)
volt_base  = multiprocessing.Array(ctypes.c_double,imsize_pb**2*nant*npol)
volt       = np.ctypeslib.as_array(volt_base.get_obj())
volt       = volt.reshape(imsize_pb,imsize_pb,nant,npol)

# construct true sky model image for extended source comprising multiple point sources
# first, make component list, then transfer to a model image
# finally, if max(nchan)>1 (ie wideband), also constuct a mask to avoid injected
# sources when calculating image statistics
def makeComponentList():
    clname = prefix+'.cl'
    os.system('rm -rf '+clname)
    cl.done()
    c = []
    c.append(c_transit.directional_offset_by(90*u.deg,0.5*las0/2*u.deg))
    c.append(c_transit)
    c.append(c_transit.directional_offset_by(-90*u.deg,0.5*las0/2*u.deg))
    fluxdensity = 1.0
    fdsum       = 0
    for comp in range(len(c)):
        ra     = c[comp].ra.to_string(u.hour)
        dec    = c[comp].dec.to_string(u.deg)
        mydir  = 'J2000 '+ra+' '+dec
        cl.addcomponent(dir=mydir,flux=fluxdensity,fluxunit='Jy',freq=str(freq)+'GHz',
                        shape='point',spectrumtype='spectral index',index=spindx);
        fdsum += fluxdensity
    
    cl.rename(filename=clname)
    cl.done()
    return c,fdsum

def makeClnMask(c,perrindx):
    # constuct mask for improved cleaning
    # circles about injected source locations with minimum radius
    # given by the larger of 2*cell_pb and perr/sqrt(Nant)
    clnmask = prefix+'_perr-'+str(perr[perrindx])+'arcsec.crtf'
    minrad  = max(2*cell_pb*3600,perr[perrindx]/np.sqrt(nant))
    f = open(clnmask,'w')
    f.write('#CRTFv0\n')
    for comp in range(len(c)):
        ra     = c[comp].ra.to_string(u.hour)
        dec    = c[comp].dec.to_string(u.deg)
        f.write('circle[['+ra+', '+dec+'], '+str(minrad)+'arcsec]\n')
    
    f.close()

# set up vp table (needed by tclean in makeImg)
# unblocked airy disk, no squint
# set maxrad to 5*HPBW at (central) freq
def makevptab():
    vptab = telname+'_vp.tab'
    os.system('rm -rf '+vptab)
    mymaxrad = '{:.2f}'.format(5*hpbw0)+'deg'
    vp.setpbairy(telescope='OTHER',othertelescope=telname,dishdiam=str(antdia)+'m',
                 blockagediam='0m',maxrad=mymaxrad,reffreq=str(freq)+'GHz')
    vp.saveastable(vptab)
    return vptab

def pntCntrs(numpt):
    # define pointing centers
    c = []
    if numpt == 3:
        #      1
        #      x         mosaic centered on x, located HPBW/sqrt(6) from pointing centers
        #     2 3
        c.append(c_transit.directional_offset_by(0*u.deg,hpbw0/np.sqrt(6)*u.deg))   # 1
        c.append(c_transit.directional_offset_by(120*u.deg,hpbw0/np.sqrt(6)*u.deg)) # 2
        c.append(c_transit.directional_offset_by(240*u.deg,hpbw0/np.sqrt(6)*u.deg)) # 3
    elif numpt == 7:
        #     2 7
        #    3 1 6       mosaic centered on 1, pointings separated by HPBW/sqrt(2)
        #     4 5
        c.append(c_transit)                                                         # 1
        c.append(c_transit.directional_offset_by(30*u.deg,hpbw0/np.sqrt(2)*u.deg))  # 2
        c.append(c_transit.directional_offset_by(90*u.deg,hpbw0/np.sqrt(2)*u.deg))  # 3
        c.append(c_transit.directional_offset_by(150*u.deg,hpbw0/np.sqrt(2)*u.deg)) # 4
        c.append(c_transit.directional_offset_by(210*u.deg,hpbw0/np.sqrt(2)*u.deg)) # 5
        c.append(c_transit.directional_offset_by(270*u.deg,hpbw0/np.sqrt(2)*u.deg)) # 6
        c.append(c_transit.directional_offset_by(330*u.deg,hpbw0/np.sqrt(2)*u.deg)) # 7
    
    return c

# construct empty MS
def makeMSFrame(nchanindx,pointingsindx,perrindx):
    numchan = nchan[nchanindx]
    numpt   = pointings[pointingsindx]
    msname1 = prefix+'_nchan-'+str(numchan)+'_perr-'+str(perr[perrindx])+\
              'asec_numpt-'+str(numpt)+'_TEMP1.ms'
    os.system('rm -rf '+msname1)
    sm.open(ms=msname1);
    sm.setconfig(telescopename=telname,x=x,y=y,z=z,dishdiameter=d,
                 mount=['alt-az'],antname=antname,coordsystem='global',
                 referencelocation=obscoord);
    sm.setfeed(mode='perfect X Y', pol=['']);
    # put all channels into a single spw over the whole band
    # freq_ch0_str is lower edge of lowest channel
    freq_ch0_str = str(freq-numchan/2*chanbw/1e3)+'GHz'
    sm.setspwindow(spwname=spwname,freq=freq_ch0_str,deltafreq=str(chanbw)+'MHz',
                   nchannels=numchan,stokes='XX YY');
    c = pntCntrs(numpt)
    for p in range(numpt):
        sm.setfield(sourcename='pointing-'+str(p+1),
                    sourcedirection=me.direction(rf='J2000',
                    v0=c[p].ra.to_string(u.hour),v1=c[p].dec.to_string(u.deg)));
    
    sm.setlimits(shadowlimit=shadowlimit,elevationlimit=elevationlimit);
    sm.setauto(autocorrwt=0.0);
    # set usehourangle=True so that starttime/stoptime will be interpreted as startha/stopha
    # hmm, don't use that here because we have multiple pointings
    # insetad, get reference time at zero hour angle when looking at c_transit and
    # set all times relative to that
    me.done()
    me.doframe(obscoord)
    # UTC (me.measure returns days)
    myreftime = me.measure(me.epoch('LAST','2040/1/1/00:00:00'),'UTC')
    #qa.time(qa.quantity(myreftime['m0']['value'],unitname='d'),form='dmy')
    # nothing special about the epoch selected above --> 24-Jul-2021/11:01:00
    sm.settimes(integrationtime=str(tint)+'s',usehourangle=False,
                referencetime=myreftime);
    # observe each pointing for 1 min
    # assume that rules for input variables have been followed
    obs0    = -obsdur/2
    tint_pt = 1  # hard-coded time(mins)/pnt; beware input variable constraints if changing
    for pobsnum in range(int(obsdur/numpt)):
        for p in range(numpt):
            # call once per pointing ("source"); for each p, construct single scan and ddid
            pstart = (obs0 + pobsnum*numpt*tint_pt + p*tint_pt)/60
            pend   = pstart + tint_pt/60
            sm.observe(sourcename='pointing-'+str(p+1),spwname=spwname,
                       starttime=str(pstart)+'h',stoptime=str(pend)+'h');
    
    sm.close();
    # flagdata(vis=msname1,mode='unflag')  # commented in case elevation/shadow flags matter

# transfer component list into model images for all pointing directions
# also create model image consistent with output mosaic image, for fidelity metric later
def makeModelImages(nchanindx,pointingsindx):
    numchan = nchan[nchanindx]
    numpt   = pointings[pointingsindx]
    c       = pntCntrs(numpt)
    clname  = prefix+'.cl'
    #for p in range(numpt):
    #    imname = prefix+'_nchan-'+str(numchan)+'_numpt-'+str(numpt)+'_pt-'+str(p)+\
    #             '.model.im'
    #    # first, create empty image, then populate
    #    ia.close()
    #    ia.fromshape(imname,[imsize_pb,imsize_pb,1,numchan],overwrite=True)
    #    cs = ia.coordsys()
    #    cs.setunits(['rad','rad','','Hz'])
    #    cs.setincrement([np.deg2rad(-cell_pb),np.deg2rad(cell_pb)],'direction')
    #    cs.setreferencevalue([c[p].ra.to_string(u.rad),
    #                          c[p].dec.to_string(u.rad)],type='direction')
    #    cs.setreferencevalue(str(freq)+'GHz','spectral')
    #    cs.setreferencepixel([(numchan-1)/2],'spectral')
    #    cs.setincrement(str(chanbw)+'MHz','spectral')
    #    ia.setcoordsys(cs.torecord())
    #    ia.setbrightnessunit('Jy/pixel')
    #    ia.set(0.0)
    #    ia.close()
    #    # populate model image with component list
    #    cl.open(clname)
    #    ia.open(imname)
    #    ia.modify(cl.torecord(),subtract=False)
    #    ia.close()
    #    cl.done()
    #    if testing: exportfits(imagename=imname,fitsimage=imname.replace('.im','.fits'))             ############ uncomment all
    
    # create model with same image parameters as later image mosaic
    # for numchan>1 only create a model image with 1 channel at freq
    imname = prefix+'_nchan-'+str(numchan)+'_numpt-'+str(numpt)+'_mosaic.model.im'
    ia.close()
    ia.fromshape(imname,[imsize_pb,imsize_pb,1,1],overwrite=True)
    cs = ia.coordsys()
    cs.setunits(['rad','rad','','Hz'])
    cs.setincrement([np.deg2rad(-cell_pb),np.deg2rad(cell_pb)],'direction')
    cs.setreferencevalue([c_transit.ra.to_string(u.rad),
                          c_transit.dec.to_string(u.rad)],type='direction')
    cs.setreferencevalue(str(freq)+'GHz','spectral')
    cs.setreferencepixel([0],'spectral')
    cs.setincrement(str(chanbw)+'MHz','spectral')
    ia.setcoordsys(cs.torecord())
    ia.setbrightnessunit('Jy/pixel')
    ia.set(0.0)
    ia.close()
    cl.open(clname)
    ia.open(imname)
    ia.modify(cl.torecord(),subtract=False)
    ia.close()
    cl.done()
    if testing: exportfits(imagename=imname,fitsimage=imname.replace('.im','.fits'))

# function to construct template images to later store antenna/baseline primary beams
# (legacy: and antenna aperture illumination functions) for all pointing directions
# note: it isn't necessary to save PBs (legacy: or AIFs) in this code, but they
#       can be useful to enable easy inspection
def makeTemplateImgs(nchanindx,pointingsindx):
    numchan = nchan[nchanindx]
    numpt   = pointings[pointingsindx]
    c       = pntCntrs(numpt)
    imname  = []
    csys    = []
    for p in range(numpt):
        imname.append(prefix+'_nchan-'+str(numchan)+'_numpt-'+str(numpt)+'_pt-'+str(p)+\
                      '.template-aif-pb.im')
        ia.close()
        ia.fromshape(imname[p],[imsize_pb,imsize_pb,1,numchan],overwrite=True)
        cs = ia.coordsys()
        cs.setunits(['rad','rad','','Hz'])
        cs.setincrement([np.deg2rad(-cell_pb),np.deg2rad(cell_pb)],'direction')
        cs.setreferencevalue([c[p].ra.to_string(u.rad),
                              c[p].dec.to_string(u.rad)],type='direction')
        cs.setreferencevalue(str(freq)+'GHz','spectral')
        cs.setreferencepixel([(numchan-1)/2],'spectral')
        cs.setincrement(str(chanbw)+'MHz','spectral')
        ia.setcoordsys(cs.torecord())
        ia.setbrightnessunit('Jy/pixel')
        ia.set(0.0)
        ia.close()
        csys.append(cs.torecord())
    
    return imname,csys

def arrayToImg(arr,outf,coord,onlyfits):
    # arr must be [imsize,imsize,stokes=1,nchan]
    myia = image()
    im   = myia.newimagefromarray(outfile=outf,pixels=arr,csys=coord,
                                  linear=False,overwrite=True)
    myia.close()
    im.close()
    if onlyfits:
        exportfits(imagename=outf,fitsimage=outf.replace('.im','.fits'),
                   overwrite=True);
        os.system('rm -rf '+outf)    # only keep fits image

def arrayFromImg(fname,fromfits):
    # arr must be [imsize,imsize,stokes=1,nchan]
    if fromfits:
        importfits(fitsimage=fname.replace('.im','.fits'),
                   imagename=fname,defaultaxes=True,
                   defaultaxesvalues=['ra','dec','stokes','freq']);
    
    myia = image()
    myia.open(fname)
    arr  = myia.getregion()
    myia.close()
    if fromfits: os.system('rm -rf '+fname)  # only keep fits image
    return arr

# legacy:
#def mydft(arr):
#    return fftpack.ifftshift(fftpack.fft2(fftpack.fftshift(arr)))

def calcParang(msname):
    # calculate parallactic angle per ant/integration, in radians
    # Note: I cannot figure out why plotms (v560) returns slightly different
    #       hour angles, and in turn slightly different parallactic angles.
    #       I checked issues with obscoord which plotms draws upon using
    #       TELESCOPE_NAME in the OBSERVATORY subtable.  Negligible impact
    #       on this simulation, so move along...
    
    # return array containing spherical latitude of each antenna, in rad
    #lat = np.deg2rad(ITRS(x,y,z).spherical.lat.value)
    # can use above, but can also get it directly from MS as follows
    
    tb.open(msname+'/ANTENNA')
    pos   = tb.getcol('POSITION')
    frame = tb.getcolkeyword('POSITION','MEASINFO')['Ref']
    units = tb.getcolkeyword('POSITION','QuantumUnits')
    tb.close()
    
    tb.open(msname)
    t0   = tb.getcol('TIME')                   # nant*(nant-1)/2 entries per unique time
    indx = np.unique(t0,return_index=True)[1]
    t    = t0[indx]                            # UTC seconds
    fid  = tb.getcol('FIELD_ID')[indx]         # pointing ID
    tb.close()
    
    tb.open(msname+'/FIELD')
    pointing_coords = tb.getcol('PHASE_DIR')
    tb.close()
    
    ra  = pointing_coords[0][0]                # radians
    dec = pointing_coords[1][0]                # radians
    
    lent   = len(t)
    parang = np.zeros([lent,nant])             # [time,ant]
    me.done()
    for n in range(lent):
        tm = me.epoch('UTC',str(t[n])+'s')
        for a in range(nant):
            antpos = me.position(frame,str(pos[0,a])+units[0],
                                       str(pos[1,a])+units[1],
                                       str(pos[2,a])+units[2])
            me.doframe(antpos)
            # spherical coordinate latitude, not geodetic
            lat = me.measure(antpos,'ITRF')['m1']['value']             # radians
            # local apparent sidereal time, radians
            # me.measure returns days
            last = (me.measure(tm,'LAST')['m0']['value']%1)*2*np.pi    # radians
            if last > np.pi: last -= 2*np.pi     # approximation, ok for obsdur<12h
            ha   = last - ra[fid[n]]                                   # radians
            parang[n,a] = np.arctan2(np.sin(ha),
                          (np.cos(dec[fid[n]])*np.tan(lat) -
                           np.sin(dec[fid[n]])*np.cos(ha)))
            #print ('t: '+str(n)+' ha: '+str(np.rad2deg(ha)/15)+\
            #       ' pa: '+str(np.rad2deg(parang[n,a])))
    
    # returns parang=[time,ant]
    # also return t and fid
    return parang,t,fid

def simVis(msname,a1,a2,c,fld,tstart,tend,imsize_pb,cell_pb,
           tempimg,vptab,framename,t_ra,t_dec):
    # Ideally don't use the im.ft approach below because that implements
    # gridder='standard' rather than 'mosaic' as used later in this code
    # for imaging.  In principle it is better to keep things consistent.
    # However, the tclean approach is 10 times slower, and it turns out
    # that it doesn't make any significant different to resulting dynamic
    # range or fidelity limits.  But beware of this difference if you
    # are modifying this code in the future. An additional factor to
    # consider if you reinstate tclean below is that you need to be
    # careful with cpuCount.  You will need to set it to a fraction
    # of total ncores because each thread will run tclean which in
    # turn will attempt to utilize all ncores in simVis().  If you
    # set cpuCount too close to ncores then you will see an efficiency
    # dropoff from read/write dominating the processing, as well as
    # as risking dropped threads or a frozen OS.  This can be solved if
    # you figure out how to force tclean to utilize only a single thread.
    
    ## simulate visibility to MODEL column
    #tclean(vis=msname,startmodel=tempimg,imagename=tempimg.replace('.im',''),
    #       savemodel='modelcolumn',gridder='mosaic',stokes='I',spw='0:'+str(c),
    #       antenna=str(a1).zfill(2)+'&&'+str(a2).zfill(2),field=str(fld),
    #       timerange=tstart+'~'+tend,vptable=vptab,imsize=imsize_pb,
    #       cell=str(cell_pb*3600)+'arcsec',weighting='natural',niter=0,
    #       normtype='flatnoise',calcres=False,calcpsf=True);
    
    im.open(msname,usescratch=True)   # ft can only write to MODEL
    im.selectvis(baseline=str(a1).zfill(2)+'&&'+str(a2).zfill(2),
                 nchan=1,start=c,field=fld,time=[tstart+'~'+tend])
    # pass characteristics of tempimg to im
    # (apparently im.setoptions won't gather this info from tempimg)
    im.defineimage(stokes='I',mode='channel',nx=imsize_pb,ny=imsize_pb,
                   cellx=str(cell_pb*3600)+'arcsec',celly=str(cell_pb*3600)+'arcsec',
                   phasecenter=me.direction(framename,v0=t_ra,v1=t_dec))
    im.setoptions(ftmachine='ft')
    im.ft(model=tempimg,incremental=False)
    im.done()

# legacy:
#def parallel_perAnt(t,c,pnterrors,parang,aif0c,testing,testingT,prefix,
#                    numchan,perr,perrindx,numpt,templateImg,templateImgCoord,
#                    aifdir,imsize_pb,pbAdir,a,def_param=volt):
#    casalog.filter('SEVERE')
#    # only corrupt phases here
#    # apply associated amplitude loss later (per baseline)
#    # assume that X feed is aligned along meridian at zero parallactic
#    # angle, ie at zero parallactic angle an X-axis offset is a V shift
#    
#    # add phase gradients in sky-frame U and V arising from pointing offset
#    # parang is orientation of sky seen from antenna
#    # so rotate by negative parang to get pointing offset seen from sky
#    shift   = pnterrors[t,a] * np.exp(-1j*parang[t,a])
#    # extra factor of 2 needed below (error in EVLA Memo 84 Equation 2)
#    phase_u = -uvals0 * -shift.imag * 2*np.pi   # +u points +W
#    phase_v = -vvals0 *  shift.real * 2*np.pi   # +v points +N
#    phase   = np.exp(1j*(phase_u+phase_v))
#    
#    # same effect for both polarizations so only need to store once
#    # perform rot90 step so we are in python image coordinates
#    # ie horizontal in output image needs to be vertical in array
#    aif           = np.rot90(aif0c * phase)
#    # complex voltage pattern is FT of aif
#    volt[:,:,a,0] = mydft(aif)
#    
#    if (testing) and (t==testingT):
#        # phase-perturbed time/ant/freq-dependent pol-independent AIFs
#        # ignore the dummy coordinates that will come with these fits images
#        # the cells should really be in units of wavelength with
#        # pixel size = cell_uv and pixel range from -uvmax to +uvmax
#        # add extra dimensions to aif for export to image (npol,nchan)
#        outf = prefix+'_nchan-'+str(numchan)+'_perr-'+str(perr[perrindx])+\
#               '_numpt-'+str(numpt)+'_tstep-'+str(testingT)+'_ant-'+str(a+1)+\
#               '_phs-perturbed_aif-phs.im'
#        if c==0:
#            os.system('cp -r '+templateImg+' '+aifdir+'/'+outf)
#            temparr = np.zeros((imsize_pb,imsize_pb,1,numchan))
#        else:
#            temparr = arrayFromImg(aifdir+'/'+outf,True)
#        
#        temparr2 = np.angle(aif,deg=True)
#        temparr2[aif==0] = 0
#        temparr[:,:,0,c] = temparr2
#        arrayToImg(temparr,aifdir+'/'+outf,templateImgCoord,True)
#        
#        # phase-perturbed time/ant/freq-dependent pol-independent PBs
#        outf = prefix+'_nchan-'+str(numchan)+'_perr-'+\
#               str(perr[perrindx])+'_numpt-'+str(numpt)+\
#               '_tstep-'+str(testingT)+'_ant-'+str(a+1)+\
#               '_phs-perturbed_pb.im'
#        if c==0:
#            os.system('cp -r '+templateImg+' '+pbAdir+'/'+outf)
#            temparr = np.zeros((imsize_pb,imsize_pb,1,numchan))
#        else:
#            temparr = arrayFromImg(pbAdir+'/'+outf,True)
#        
#        # square voltage amplitude to get power,
#        # then normalize to unity for this
#        temparr[:,:,0,c]  = np.abs(volt[:,:,a,0])**2
#        temparr[:,:,0,c] /= np.max(temparr[:,:,0,c])
#        arrayToImg(temparr,pbAdir+'/'+outf,templateImgCoord,True)

# make voltage pattern
# this is an alternate method (more accurate) to avoid FT'ing AIFs
# former is only possible if the antenna voltage patterns can be well
# described by an analytic function
# Here this is ok because we want unblocked apertures without squint,
# in which case the voltage pattern is simply an airy disk
# Note that the FT AIF approach works, but in practice it requires
# imsize_pb much larger than 4096 to sample the aperture with more than
# a dozen pixels.  If only a dozen pixels are used, these will introduce
# pixelization errors that will in turn limit numerical dynamic range
def makeVolt(lam,antdia,shift_u,shift_v):
    # radius in radians, add negligible 1e-50 so we can never have radius=0
    radius  = np.sqrt((uvals0-shift_u)**2 + (vvals0-shift_v)**2) + 1e-50
    temparr = 2*np.pi/lam*antdia/2*np.sin(radius)
    return np.abs(2*j1(temparr)/temparr)

def parallel_perAnt(t,c,lam,pnterrors,parang,testing,testingT,prefix,
                    numchan,perr,perrindx,numpt,templateImg,templateImgCoord,
                    imsize_pb,pbAdir,antdia,a,def_param=(volt,uvals0,vvals0)):
    casalog.filter('SEVERE')
    # only corrupt phases here (ie PB positions)
    # apply associated amplitude loss later (per baseline)
    # assume that X feed is aligned along meridian at zero parallactic
    # angle, ie at zero parallactic angle an X-axis offset is a V shift
    
    # add position offsets in sky-frame U and V arising from pointing offset
    # parang is orientation of sky seen from antenna
    # so rotate by negative parang to get pointing offset seen from sky
    shift   = pnterrors[t,a] * np.exp(-1j*parang[t,a])
    shift_u = shift.imag    # +u points +E, in radians
    shift_v = shift.real    # +v points +N, in radians
    
    # same effect for both polarizations so only need to store once
    volt[:,:,a,0] = makeVolt(lam,antdia,shift_u,shift_v)
    
    if (testing) and (t==testingT):
        # phase-perturbed time/ant/freq-dependent pol-independent PBs
        outf = prefix+'_nchan-'+str(numchan)+'_perr-'+\
               str(perr[perrindx])+'_numpt-'+str(numpt)+\
               '_tstep-'+str(testingT)+'_ant-'+str(a+1)+\
               '_phs-perturbed_pb.im'
        if c==0:
            os.system('cp -r '+templateImg+' '+pbAdir+'/'+outf)
            temparr = np.zeros((imsize_pb,imsize_pb,1,numchan))
        else:
            temparr = arrayFromImg(pbAdir+'/'+outf,True)
        
        # square voltage amplitude to get power
        # result is already normalized to unity
        temparr[:,:,0,c] = (volt[:,:,a,0])**2
        arrayToImg(temparr,pbAdir+'/'+outf,templateImgCoord,True)

def parallel_perBl(t,c,pnterrors,parang,sigma,vptab,testing,testingT,prefix,
                   numchan,perr,perrindx,numpt,templateImg,templateImgCoord,
                   pbBdir,imsize_pb,timestamp,tint,fid,cell_pb,c_transit,
                   msname_bl,def_param=(volt,uvals0,vvals0)):
    casalog.filter('SEVERE')
    
    ### construct sky-frame perturbed baseline-dependent primary
    ### beams at this time/freq (polarization-independent)
    
    a1 = int(re.split('[_ .]',msname_bl)[-3])
    a2 = int(re.split('[_ .]',msname_bl)[-2])
    
    # baseline power pattern = FT(convolved aperture functions)
    #                        = multiplied FT(voltage pattern)
    # ie bpb = FT(a1*conj(a2)) = FT(a1) . FT(a2)
    bpb     = volt[:,:,a1,0] * volt[:,:,a2,0]
    # amplitude loss associated with differential pointing
    # between antennas
    perr1   = pnterrors[t,a1] * np.exp(-1j*parang[t,a1])
    perr2   = pnterrors[t,a2] * np.exp(-1j*parang[t,a2])
    attnAmp = np.exp(-(np.abs(perr1-perr2))**2/(4*sigma**2))
    # normalize amplitude to attenuated level
    #bpb    *= attnAmp / np.max(np.abs(bpb))   # legacy
    bpb    *= attnAmp / np.max(bpb)
    
    if (testing) and (t==testingT):
        # amp+phase-perturbed time/ant/freq-dependent
        # pol-independent baseline PBs
        outf = prefix+'_nchan-'+str(numchan)+'_perr-'+\
               str(perr[perrindx])+'_numpt-'+str(numpt)+\
               '_tstep-'+str(testingT)+'_ant1-'+str(a1+1)+\
               '_ant2-'+str(a2+1)+'_amp-phs-perturbed_bpb.im'
        if c==0:
            os.system('cp -r '+templateImg+' '+pbBdir+'/'+outf)
            temparr = np.zeros((imsize_pb,imsize_pb,1,numchan))
        else:
            temparr = arrayFromImg(pbBdir+'/'+outf,True)
        
        #temparr[:,:,0,c] = np.abs(bpb)        # legacy
        temparr[:,:,0,c] = bpb
        arrayToImg(temparr,pbBdir+'/'+outf,templateImgCoord,True)
        del temparr
    
    ### construct true sky image seen by this baseline
    ### (polarization-independent)
    
    temparr = arrayFromImg(prefix+'_nchan-'+str(numchan)+'_numpt-'+\
                           str(numpt)+'_pt-'+str(fid)+'.model.im',False)
    tempimg = prefix+'_nchan-'+str(numchan)+'_perr-'+str(perr[perrindx])+\
              '_numpt-'+str(numpt)+'_tstep-'+str(t)+'_ant1-'+str(a1+1)+\
              '_ant2-'+str(a2+1)+'_temp.im'
    #temparr[:,:,0,c] *= np.abs(bpb)           # legacy
    temparr[:,:,0,c] *= bpb
    arrayToImg(temparr,tempimg,templateImgCoord,False)
    
    ### simulate visibilities from the corrupted sky model into
    ### the MODEL column for all integrations in this timestep
    
    tstart = qa.time(qa.quantity(timestamp-tint/2,unitname='s'),form='ymd')[0]
    tend   = qa.time(qa.quantity(timestamp+tint/2,unitname='s'),form='ymd')[0]
    # simVis is a time sink, which explains the gymnastics
    # with the parallelization and MSname disk space usage
    # (don't bother using the MS partition route, a-la pieflag)
    simVis(msname_bl,a1,a2,c,fid,tstart,tend,imsize_pb,cell_pb,tempimg,
           vptab,c_transit.frame.name,c_transit.ra.to_string(u.hour),
           c_transit.dec.to_string(u.deg))
    os.system('rm -rf '+tempimg.replace('.im','*'))

# corrupt visibilities
def corruptVis(nchanindx,pointingsindx,perrindx,vptab,
               templateImgs,templateImgCoords,pool):
    numchan = nchan[nchanindx]
    numpt   = pointings[pointingsindx]
    msname  = prefix+'_nchan-'+str(numchan)+'_perr-'+str(perr[perrindx])+\
              'asec_numpt-'+str(numpt)+'.ms'    # split msname1 here (data col)
    msname1 = msname.replace('.ms','_TEMP1.ms') # template, concat bl's here later
    
    # calculate parallactic angle per ant/integration, to enable later translation
    # of pointing errors from X-Y to U-V coordinate frames
    # note: src decl < obs lat here, so parang is negative as src rises in East
    # parang [time,ant] in radians
    # also returns timestamps (all integrations, UTC seconds) and fid (integer,
    # all integrations), latter needed for ft later
    parang,timestamps,fid = calcParang(msname1)
    
    # legacy:
    ## construct antenna-frame unperturbed freq-dependent ant/pol/time-independent AIFs
    
    ## construct antenna-frame unperturbed freq-dependent ant/pol/time-independent PBs
    ## assume unblocked circular aperture
    
    #aif0 = np.zeros((imsize_pb,imsize_pb,numchan),'complex')    # legacy
    sigma = np.zeros(numchan)
    lam   = np.zeros(numchan)
    for c in range(numchan):
        lam[c] = c0 / (freq*1e9 + ((1-numchan)/2+c)*chanbw*1e6)
        # legacy:
        #r     = antdia/2/lam[c] / cell_uv  # radius in wavelengths normalized by cell_uv
        #disk  = radius2 < r**2
        #aif0[disk,c] = 1+0j
        
        # ant primary beam power profile, airy disk (suitable here, simulated)
        # used later to attenuate baseline primary beams
        hpbw     = 1.028*lam[c]/antdia                   # radians
        sigma[c] = hpbw / (2*np.sqrt(2*np.log(2)))       # radians
    
    trange = range(ntsteps)
    if testing:
        if rapidtest: trange = [testingT]
        
        # legacy:
        ## unperturbed freq-dependent ant/pol/time-independent AIFs
        #outf = prefix+'_nchan-'+str(numchan)+'_perr-'+str(perr[perrindx])+\
        #       '_numpt-'+str(numpt)+'_unperturbed_aif-abs.im'
        #os.system('cp -r '+templateImgs[fid[testingT]]+' '+aifdir+'/'+outf)
        ## ignore the dummy coordinates that will come with these fits images
        ## the cells should really be in units of wavelength with
        ## pixel size = cell_uv and pixel range from -uvmax to +uvmax
        ## add extra dimension to aif0 for export to image
        #temparr = np.zeros((imsize_pb,imsize_pb,1,numchan),'complex')
        #temparr[:,:,0,:] = aif0
        #arrayToImg(np.abs(temparr),aifdir+'/'+outf,
        #           templateImgCoords[fid[testingT]],True)
        
        # unperturbed freq-dependent ant/pol/time-independent PBs
        outf = prefix+'_nchan-'+str(numchan)+'_perr-'+str(perr[perrindx])+\
               '_numpt-'+str(numpt)+'_unperturbed_pb.im'
        os.system('cp -r '+templateImgs[fid[testingT]]+' '+pbAdir+'/'+outf)
        temparr = np.zeros((imsize_pb,imsize_pb,1,numchan))
        for c in range(numchan): temparr[:,:,0,c] = (makeVolt(lam[c],antdia,0,0))**2
        
        arrayToImg(temparr,pbAdir+'/'+outf,templateImgCoords[fid[testingT]],True)
        del temparr
    
    # assign pointing errors as complex numbers for all antennas in all integrations
    # +real=+X +imag=+Y; phase 0=X, increases North through East
    # pointing error = radial offset (amp), in radians
    myscale   = np.deg2rad(perr[perrindx]/3600)/np.sqrt(np.pi/2) # rayleigh statistics
    pnterrors = np.random.normal(size=(ntsteps,nant),scale=myscale) +\
                np.random.normal(size=(ntsteps,nant),scale=myscale)*1j
    # save full output in case results from random instance needed later (unlikely?)
    outf = prefix+'_nchan-'+str(numchan)+'_perr-'+str(perr[perrindx])+\
           'asec_numpt-'+str(numpt)+'_pnterrors.pickle'
    with open(outf,'wb') as handle: pickle.dump(pnterrors,handle)
    # also save in human readable format, only export magnitudes of offsets
    np.savetxt(outf.replace('.pickle','-mag.txt'),
               np.rad2deg(np.abs(pnterrors))*3600,
               fmt='%5.1f',header='rows=timesteps columns=antennas units=arcsec')
    
    bl_list = []
    for a1 in range(nant):
        for a2 in range(nant):
            if a1 < a2:
                msname_bl = msname1.replace('.ms','_'+str(a1)+'_'+str(a2)+'.ms')
                bl_list.append(msname_bl)
                # hack MS partition
                os.system('cp -r '+msname1+' '+msname_bl)
    
    # be careful with memory management:
    #   footprint per array ~= cmplx_array.size * 15 / 1e9 GB (!)
    # to lower memory footprint, corrupt visibilities per freq/time
    # loop over channels, then integrations
    # Could add other errors below if needed in future
    for c in range(numchan):
        for t in trange:
            extra = ''
            if (testing) and (t==testingT): extra = ' (includes writing out FITS images)'
            print (' chan='+str(c+1)+'/'+str(numchan)+\
                   ' timestep='+str(t+1)+'/'+str(ntsteps)+extra)
            
            ### construct sky-frame (parang-compensated) perturbed ant/pol-dependent
            ### AIFs and voltage beams at this time/freq
            
            f = partial(parallel_perAnt,t,c,lam[c],pnterrors,parang,testing,
                        testingT,prefix,numchan,perr,perrindx,numpt,templateImgs[fid[t]],
                        templateImgCoords[fid[t]],imsize_pb,pbAdir,antdia)
            pool.map(f,range(nant))
            
            ### corrupt visibilities per baseline
            
            f = partial(parallel_perBl,t,c,pnterrors,parang,sigma[c],vptab,
                        testing,testingT,prefix,numchan,perr,perrindx,numpt,
                        templateImgs[fid[t]],templateImgCoords[fid[t]],pbBdir,
                        imsize_pb,timestamps[t],tint,fid[t],cell_pb,c_transit)
            pool.map(f,bl_list)
    
    for a1 in range(nant):
        for a2 in range(nant):
            if a1 < a2:
                msname_bl = msname1.replace('.ms','_'+str(a1)+'_'+str(a2)+'.ms')
                ms.open(msname_bl)
                ms.msselect({'baseline':str(a1)+'&&'+str(a2)})
                temparr = ms.getdata('model_data')
                ms.close()
                os.system('rm -rf '+msname_bl)
                ms.open(msname1,nomodify=False)
                ms.msselect({'baseline':str(a1)+'&&'+str(a2)})
                ms.putdata(temparr)
                ms.close()
    
    split(vis=msname1,outputvis=msname,datacolumn='model')
    os.system('rm -rf '+msname1)

# image corrupted MS, use same image frame as defined earlier for inputs
def makeImg(nchanindx,pointingsindx,perrindx,vptab):
    print (' imaging with clean fractional threshold = '+str(cleanfrac[perrindx]))
    numchan = nchan[nchanindx]
    clnmask = prefix+'_perr-'+str(perr[perrindx])+'arcsec.crtf'
    msname  = prefix+'_nchan-'+str(numchan)+'_perr-'+str(perr[perrindx])+\
              'asec_numpt-'+str(pointings[pointingsindx])+'.ms'
    imgdir  = msname.replace('.ms','')
    os.system('rm -rf '+imgdir)
    os.mkdir(imgdir)
    os.chdir(imgdir)
    myphasecenter     = me.measure(me.direction(c_transit.frame.name,
                                   v0=c_transit.ra.to_string(u.hour),
                                   v1=c_transit.dec.to_string(u.deg)),'J2000')
    myphasecenter_str = 'J2000 ' + qa.time(qa.quantity(myphasecenter['m0']['value'],
                                           unitname='rad'))[0] + ' ' +\
                                   qa.angle(qa.quantity(myphasecenter['m1']['value'],
                                           unitname='rad'))[0]
    
    trueskyJYPIX    = '../'+prefix+'_nchan-'+str(numchan)+'_numpt-'+\
                      str(pointings[pointingsindx])+'_mosaic.model.im'
    trueskyJYPIXpba = 'sim.truesky.pbattn'
    trueskyJYBEAM   = 'sim.truesky.pbattn.conv'
    ia.open(trueskyJYPIX)
    truesky_cs      = ia.coordsys()
    truesky_shape   = ia.shape()
    ia.close()
    
    if numchan == 1:
        # dirty image
        tclean(vis='../'+msname,imagename='sim',imsize=imsize_pb,
               cell=str(cell_pb*3600)+'arcsec',gridder='mosaic',normtype='flatnoise',
               vptable='../'+vptab,phasecenter=myphasecenter_str,
               weighting='natural',niter=0);
        mystats = imstat(imagename='sim.residual',listit=False,verbose=False);
        mythreshold = mystats['max'][0] * cleanfrac[perrindx]
        os.system('cp -r sim.image sim.image.dirty')
        exportfits(imagename='sim.image',fitsimage='sim.image.dirty.fits')                    ############## comment
        exportfits(imagename='sim.psf',fitsimage='sim.psf.fits')                            ############## comment
        casalog.filter('INFO')   # reinstate info for imaging (after dirty image)
        # cleaned image
        tclean(vis='../'+msname,imagename='sim',imsize=imsize_pb,
               cell=str(cell_pb*3600)+'arcsec',gridder='mosaic',normtype='flatnoise',
               vptable='../'+vptab,phasecenter=myphasecenter_str,
               weighting='natural',niter=1000,mask=['../'+clnmask],
               threshold=mythreshold,calcres=False,calcpsf=False);
        casalog.filter('SEVERE')
        #exportfits(imagename='sim.image',fitsimage='sim.image.fits')
        #exportfits(imagename='sim.residual',fitsimage='sim.residual.fits')
        #exportfits(imagename='sim.model',fitsimage='sim.model.fits')
        exportfits(imagename='sim.pb',fitsimage='sim.pb.fits')                                  ############## comment
        psf_im = 'sim.psf'
        # PB-attenuate true sky model
        immath(imagename=[trueskyJYPIX,'sim.pb'],outfile=trueskyJYPIXpba,
               expr='IM0*IM1')
    else:
        # dirty image
        tclean(vis='../'+msname,imagename='sim',imsize=imsize_pb,
               cell=str(cell_pb*3600)+'arcsec',gridder='mosaic',normtype='flatnoise',
               vptable='../'+vptab,phasecenter=myphasecenter_str,
               weighting='natural',deconvolver='mtmfs',nterms=nterms,niter=0);
        mystats = imstat(imagename='sim.residual.tt0',listit=False,verbose=False);
        mythreshold = mystats['max'][0] * cleanfrac[perrindx]
        os.system('cp -r sim.image.tt0 sim.image.tt0.dirty')
        #exportfits(imagename='sim.image.tt0',fitsimage='sim.image.tt0.dirty.fits')
        #exportfits(imagename='sim.psf.tt0',fitsimage='sim.psf.tt0.fits')
        casalog.filter('INFO')   # reinstate info for imaging (after dirty image)
        # cleaned image
        tclean(vis='../'+msname,imagename='sim',imsize=imsize_pb,
               cell=str(cell_pb*3600)+'arcsec',gridder='mosaic',normtype='flatnoise',
               vptable='../'+vptab,phasecenter=myphasecenter_str,
               weighting='natural',deconvolver='mtmfs',nterms=nterms,
               niter=1000,mask=['../'+clnmask],threshold=mythreshold,
               calcres=False,calcpsf=False);
        casalog.filter('SEVERE')
        #exportfits(imagename='sim.image.tt0',fitsimage='sim.image.tt0.fits')
        #exportfits(imagename='sim.residual.tt0',fitsimage='sim.residual.tt0.fits')
        #exportfits(imagename='sim.model.tt0',fitsimage='sim.model.tt0.fits')
        #exportfits(imagename='sim.pb.tt0',fitsimage='sim.pb.tt0.fits')
        psf_im = 'sim.psf.tt0'
        # PB-attenuate true sky model
        immath(imagename=[trueskyJYPIX,'sim.pb.tt0'],outfile=trueskyJYPIXpba,
               expr='IM0*IM1')
    
    # convolve PB-attenuated true sky model with PSF, for fidelity metric later
    ia.open(trueskyJYPIXpba)
    truesky_im=ia.convolve(outfile=trueskyJYBEAM,kernel=psf_im,scale=1.0)
    truesky_im.setbrightnessunit('Jy/beam')
    truesky_im.done()
    ia.close()
    exportfits(imagename=trueskyJYBEAM,fitsimage=trueskyJYBEAM+'.fits')                            ############## comment
    os.chdir('../')

# capture image dynamic range and fidelity statistics
def calcStats(nchanindx,pointingsindx,perrindx,fdsum):
    print (' recording statistics')
    numchan = nchan[nchanindx]
    imgdir  = prefix+'_nchan-'+str(numchan)+'_perr-'+str(perr[perrindx])+\
              'asec_numpt-'+str(pointings[pointingsindx])
    if numchan == 1:
        ia.open(imgdir+'/sim.image')
        img_max   = ia.statistics()['max'][0]
        ia.close()
        ia.open(imgdir+'/sim.residual')
        res_rms   = ia.statistics()['rms'][0]
        ia.close()
        ia.open(imgdir+'/sim.pb')
        pbmask    = np.squeeze(ia.getchunk()).flatten()
        pbmask    = np.where(pbmask<pbthresh,0,1)
        ia.close()
        ia.open(imgdir+'/sim.image.dirty')
        dimgvals = np.squeeze(ia.getchunk()).flatten()
        ia.close()
    else:
        ia.open(imgdir+'/sim.image.tt0')
        img_max = ia.statistics()['max'][0]
        ia.close()
        ia.open(imgdir+'/sim.image.tt0')
        # std = 1.48 * MAD for normally distributed data
        # this is not the same as rms, but should be approximately the same
        # for this simulation with expected zero mean & median
        # Don't use rms because the few pixels near the cutoff
        # threshold (includepix) will bias the results
        res_rms = ia.statistics(includepix=0.1,
                                robust=True)['medabsdevmed'][0] * 1.48
        ia.close()
        ia.open(imgdir+'/sim.pb.tt0')
        pbmask    = np.squeeze(ia.getchunk()).flatten()
        pbmask    = np.where(pbmask<pbthresh,0,1)
        ia.close()
        ia.open(imgdir+'/sim.image.tt0.dirty')
        dimgvals = np.squeeze(ia.getchunk()).flatten()
        ia.close()
    
    dr = img_max / res_rms
    ia.open(imgdir+'/sim.truesky.pbattn.conv')
    tpbacvals = np.squeeze(ia.getchunk()).flatten()
    ia.close()
    # apply pbthresh mask
    dimgvals  = np.where(pbmask,dimgvals,0)
    tpbacvals = np.where(pbmask,tpbacvals,0)
    maxvals   = np.maximum.reduce([dimgvals,tpbacvals])
    absdeltas = np.abs(dimgvals-tpbacvals)
    imf = 1 - (maxvals*absdeltas).sum() / (maxvals*maxvals).sum()
    return (dr,imf)

def main():
    if not imgOnly: pool  = multiprocessing.Pool(cpuCount)
    casalog.filter('SEVERE')     # suppress screen warnings (imaging excepted)
    vptab = makevptab()
    cmplist,fdsum = makeComponentList()
    for k in range(len(pointings)):       # number of pointings
        for i in range(len(nchan)):       # number of channels
            if not imgOnly:
                makeModelImages(i,k)                                    ####################### move back under if not imgOnly
                templateImgs,templateImgCoords = makeTemplateImgs(i,k)
                if not testing:
                    # If testing, keep template image to later copy and store AIFs
                    # and ant/bl PBs.  Store templateImgCoord whether testing or not
                    for p in range(k): os.system('rm -rf '+templateImgs[k])
            
            f = open(prefix+'_nchan-'+str(nchan[i])+'_numpt-'+\
                     str(pointings[k])+'_results.txt','w')
            if nchan[i] > 1: f.write('# injected spectral index = '+str(spindx)+'\n')
            f.write('# perr (arcsec)    dynamic range    fidelity\n')
            for m in range(len(perr)):    # number of pointing errors
                print ('=== Processing nchan='+str(nchan[i])+', '+\
                       str(pointings[k])+'-pnt mosaic, pointing error='+\
                       str(perr[m])+'" ===')
                if not imgOnly:
                    if (i==0) and (k==0): makeClnMask(cmplist,m)
                    makeMSFrame(i,k,m)
                    corruptVis(i,k,m,vptab,templateImgs,templateImgCoords,pool)
                
                makeImg(i,k,m,vptab)
                # record dynamic range and fidelity statistics
                dr,imf = calcStats(i,k,m,fdsum)
                f.write('%10.1f %19.2e %13.2e\n' % (perr[m],dr,imf))
            
            f.close()
    
    if not imgOnly:
        pool.close()
        pool.join()

if __name__ == '__main__':
    main()

