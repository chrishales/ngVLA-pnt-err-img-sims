# C. Hales, NRAO
# version 1.0 (24 April 2020)
#
# Measure image dynamic range and fidelity achievable with different dynamic pointing
# errors for ngVLA MA revC (214 x 18m).  The procedure involves constructing mis-pointed
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
#
# requires: Python 3.6 and CASA 6.0
# https://casa.nrao.edu/casadocs/casa-5.6.0/introduction/casa6-installation-and-usage
# https://gitlab.nrao.edu/rurvashi/simulation-in-casa-6/blob/master/Simulation_Script_Demo.ipynb
#
# simulation details:
# - this is a lightly modified version of sim_pointing_SBA.py (inherits code structure)
# - natural weighted ngVLA MA revC, no beam sculpting.  This results in a PSF approximately
#      10x larger than would be expected from an array containing 1000 km baselines.
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
# - observation duration is centered about zero hour angle (avoids shadowing) with
#      transit at 10 deg from zenith
# - performed at 27 GHz for ease of code migration from sim_pointing_SBA.py
# - two spectral arrangements:
#      (i)  single narrow-band channel
#      (ii) wide fractional bandwidth (13/27 = 48%) with power law emission
#           (band 4: 20.5 - 34 GHz, max instantaneous bw 13.5 GHz; this sim samples 13 GHz)
# - single pointing, observed continuously
# - two source arrangements:
#     (i)  single point source at field center
#     (ii) single point source at half-power radius (due East)
#     For reference:
#            20 GHz: HPBW ~= 2.9', PSF ~= 0.031" (100 km max baseline, as above)
#            27 GHz: HPBW ~= 2.1', PSF ~= 0.023"
#            34 GHz: HPBW ~= 1.7', PSF ~= 0.018"
# - all antennas are assumed to have identical apertures that are circular, unblocked,
#      and with zero squint.  (Note: unlike the SBA simulation code, this code
#      does not require mosaic imaging, so there is no need to use modified CASA 6.0)
# - rather than FT phase-corrupted aperture illumination functions per antenna to produce
#      mis-pointed antenna primary beams (the relevant fully-functional code for which is
#      commented out below, in case you are curious; search for the word "legacy"), this
#      code simply offsets airy disks to produce the latter.  This allows for much greater
#      angular resolution in the primary beams and in turn fewer computational artifacts
#      that would otherwise bias the dynamic range and image fidelity measurements (see
#      additional commentary above function makeVolt).  This is possible because this
#      code assumes unblocked circular apertures with zero squint.
# - this code only works for injected models containing a single point source located
#      anywhere in the FOV.  This is necessary to enable the use of a trick in this code
#      to enable imaging at the resolution of the longest MA baseline while voltage
#      patterns are sampled at much coarser angular resolution (they do not need anywhere
#      near the long baseline resolution), where a corrupted component list is calculated
#      per time/chan/baseline.  It is conceivable that this approach could work for
#      multiple separated point sources, or even more complex arrangements, but such
#      options are beyond the scope of this version.
# - the casa logfile produced by this code only contains output from imaging.  All other
#      processing details are suppressed.  You will see some imaging warnings printed to
#      screen; these are benign, unless you see failures being reported.
# - the observatory center will be defined as the ngVLA location pre-defined in CASA,
#      rather than the mean of antenna positions (cf. sim_pointing_SBA.py)
# - imaging uses tclean with the standard gridder (no widefield A or W effects) and
#      natural weighting.  Only a small region about the injected source will be
#      imaged, rather than the full primary beam.
# - clean mask applied in tclean to prevent cleaning sidelobes, given by circle about
#      injected source locations with radius given by 2*PSF_FWHM.  The default threshold
#      assigned below is the smallest value appropriate before sidelobes will begin to
#      be cleaned.  The numerical noise in an image residual resulting from a single
#      channel zero pointing error simulation is limited by cleanfrac; a reasonable
#      limiting noise level that is appropriate for perr=10arcsec is just below 1e-4,
#      yielding a dynamic range limit around 40 dB.  Consult ngVLA Memo 60 to ensure
#      that you aren't setting parameters that will result in unachievable dynamic
#      range parameter space.
# - parallel processing using python multiprocessing, will use num_cores-2 leaving 2
#      free so your OS doesn't freeze.  Note that imaging is not parallelized here
#      (but not a major time sink; casampi is not used here).  As a guide, this code
#      can process 1 channel and 1 timestep with 1024x1024 arrays running in parallel
#      with 38 cores in around 10 mins.  However, this does not include overheads to
#      initialize or recombine data for parallelization, which takes around 60 mins
#      each (ie 2 hours total per run).  This whole process is read/write limited
#      rather than CPU-limited. This is the least-worst of several options I tested;
#      while there is probably a more efficient way to write the parallelization
#      (e.g. somehow bunching baselines into packages sent to threads), the current
#      version works in an acceptable wall-clock timeframe and is thus good enough
#      for the stated purpose of this code.
# - memory management has been considered.  I ran this code on a machine with 256 GB
#      RAM.  You may be ok with half this.  But one thing you will need is disk space.
#      The parallelization scheme implemented here can take up to 200 GB (eg with 10's
#      of timesteps, or 1 timestep and 10's of channels)
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
# - fidelity is defined in calcStats as 1 - abs(max[image_pbcor] - model_true) /
#      model_true , where model_true = 1 due to injection of a single point source
#      with unit flux density.

#########################################################################################

antloc  = 'ngvla-main-revC.cfg'  # include full path if not in cwd
                                 # Note: this simulation will assign the
                                 #       observatory location for NGVLA defined
                                 #       inside CASA

telname = 'NGVLA_MA'     # rename header from antloc
                         #    if this includes '*NGVLA*' then tasks like plotms
                         #    and listobs will assume the observatory location for
                         #    NGVLA defined inside CASA

perr = [0,10,20,40,60]   # list of pointing errors, radial rms in arcsec
freq      = 27           # central frequency of band, in GHz
spwname   = 'Band4'      # spectral window name to record in MS
chanbw    = 1000         # channel bandwidth, in MHz
                         #   (only needed for multi-channel spacing)
nchan     = [1]          # list containing number of channels across band
                         #   (odd numbers only, eg [1,13])
obsdur    = 10           # total observation duration, in minutes
tint      = 60           # integration time, in seconds
                         #   (obsdur[sec]/tint[sec] must return an integer)
srcloc    = [0]          # location of injected source, 0=on-axis, 1=off-axis
                         #   (must be [0], [1], or [0,1]; these values
                         #    reflect structure hardcoded in makeComponentList)
spindx    = -0.8         # spectral index S \propto nu**spindx for injected sources

shadowlimit    = 0.01    # antenna shadow limit
elevationlimit = '45deg' # antenna elevation limit, visibilities flagged below this

# size for primary beams (legacy: also used for aperture illumination functions)
# This code will use cell_pb to sample the effects of corrupted PBs on the
#   injected model, because there is no need to sample the PB with cell_im.
#   However, unlike sim_pointing_SBA.py, this code will not FT the corrupted
#   cell_pb image of the model, but will rather extract the resulting corrupted
#   peak flux density and compute a new component model, so that FTs for long
#   baselines are calculated accurately.
cell_pb   = 0.4/3600     # deg; ensure at least 100 pixels across HPBW
                         #    more strict: ensure a few pixels per perr
                         #   (airy disk HPBW ~= 2.9'/2.1'/1.7')
imsize_pb = 1024         # ensure that cell_pb*imsize_pb is around 2 x HPBW so that
                         #    the FOV is large enough to encompass the pointing
                         #   (even numbers only)
                         #    legacy: make cell_pb*imsize_pb as large as possible,
                         #    at least ~20x HPBW, so that the aperture illumination
                         #    functions are sampled by at least a few dozen pixels
                         #   (don't go larger than 4096, or even 2048, or else you
                         #    will probably run out of memory)
cell_im   = 0.02/10      # arcsec; ensure at least 10 pixels per PSF FWHM
                         #   (PSF FWHM ~= 0.02" in band 4, no tapering, recall header)
imsize_im = 1470         # make this as large as necessary to encompass significant
                         #    PSF sidelobe structure, otherwise the dynamic range
                         #    estimates could be biased.  Note: 2*3*5*7^2 = 1470
cleanfrac = [0.0005,0.0005,0.001,0.003,0.006]
                         # cleaning threshold as fraction of peak in dirty image,
                         #   corresponding to each perr above (independent of
                         #   nchan or pointings).  Ensure cleanfrac doesn't exceed
                         #   the dynamic range of the simulation (will require
                         #   some experimentation, set imgOnly=True to rerun
                         #   simulation with imaging only).  Note that tclean
                         #   auto-multithresh is not used here.
                         #   (DR limit of this code ~= 0.0001 for perr=0)
                         #   I used the following:
                         #   (pnts=pointings, nint=obsdur[sec]/tint[sec])
                         #   * perr=[0,10,20,40,60], nint=1, nchan=[1], on-axis
                         #     cleanfrac=[0.0005,0.0005,0.001,0.003,0.006]
                         #   * perr=[0,10,20,40,60], nint=1, nchan=[1], off-axis
                         #     cleanfrac=[0.0005,0.001,0.003,0.006,0.01]
                         #   * perr=[0,10,20,40,60], nint=10, nchan=[1], on-axis
                         #     cleanfrac=[0.0002,0.0002,0.0005,0.001,0.002]
                         #   * perr=[0,10,20,40,60], nint=10, nchan=[1], off-axis
                         #     cleanfrac=[0.0005,0.0007,0.001,0.002,0.002]
nterms    = 5            # number of taylor terms for cleaning multi-channel cases
                         #    (select >4 to ensure expansion errors don't dominate)

testing   = False        # write out fits images for the model, unperturbed
                         #   (template) ant pb (legacy: and AIF) and, for a nominated
                         #    time step only (testingT), the ant + bl pb's
                         #   (True/False)
testingT  = 4            # integration number (0-based) to write images above, noting
                         #    that parang~=0 is at testingT = int((obsdur*60/tint-1)/2)
rapidtest = False        # if True, and if testing=True, ignore obsdur and only
                         #    run the simulation at a single timestep (and single
                         #    pointing; consider cleanfrac) at timestep = testingT
imgOnly   = False        # If True, don't run simulations, only perform imaging and
                         #    write out new results.  If False, include simulations.
                         #    You must run with False prior to running with True.

#########################################################################################

import os,sys,pickle,re
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
cpuCount = os.cpu_count()-2

prefix     = 'point-source'
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

# set RA and Dec at pointing center
# assign former at transit and latter using NGVLA-phi offset by -10 deg,
# where NGVLA-phi comes from the ngVLA origin using (ITRF-based)
# me.observatory('ngVLA') but this won't return the mean SBA phi
# If you use the following it will assign the obscoord that is pre-defined
# inside CASA for the telname in the header (ie NGVLA)
tempdict = sl.readantenna(antloc)
# sl.readantenna built from source returns 8 keys, while pre-built returns 7, yay
if len(tempdict) == 7:
    (x,y,z,d,antname,telnameDUMMY,obscoord) = tempdict
elif len(tempdict) == 8:
    (x,y,z,d,padname,antname,telnameDUMMY,obscoord) = tempdict
else:
    print ('  ERROR - Unknown format returned by sl.readantenna.  Exiting.')
    sys.exit()

del telnameDUMMY    # use the value defined earlier, everything else will work fine
# old code from sim_pointing_SBA.py which assigns obscoord using mean position:
## Instead, take the antenna coordinates and assign our own observatory location
#x          = np.loadtxt(antloc,usecols=0)
#y          = np.loadtxt(antloc,usecols=1)
#z          = np.loadtxt(antloc,usecols=2)
#d          = np.loadtxt(antloc,usecols=3,dtype=int)
#antname    = np.loadtxt(antloc,usecols=4,dtype=str)
## assign observatory location at mean antenna position
#obscoord   = me.position('ITRF',str(np.mean(x))+'m',
#                                str(np.mean(y))+'m',
#                                str(np.mean(z))+'m')
nant       = len(antname)
antdia     = d[0]                     # all same, in meters
phi_mean   = np.rad2deg(ITRS(x,y,z).spherical.mean().lat.value)
ra_transit = Angle(0,u.hourangle)     # center sources at RA = 0
                                      #  (observe HA = -obsdur/2 --> +obsdur/2)
decl       = Angle(phi_mean-10,u.deg) # keep dec constant for all injected gaussians
c_transit  = SkyCoord(ra_transit,decl,frame='icrs')
ntsteps    = np.int(obsdur*60/tint)   # total number of time steps
minsep     = 1e10
maxsep     = 0
for a1 in range(nant):
    for a2 in range(nant):
        if a1 < a2:
            sep = np.sqrt((x[a1]-x[a2])**2+(y[a1]-y[a2])**2+(z[a1]-z[a2])**2)
            if sep < minsep: minsep = sep
            if sep > maxsep: maxsep = sep

# all in deg
psf0  = np.rad2deg(c0/(freq*1e9)/maxsep)*10  # empirical x10 (full MA imaged, see header)
las0  = np.rad2deg(c0/(freq*1e9)/minsep)
hpbw0 = np.rad2deg(c0/(freq*1e9)/antdia)

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

# construct true sky model images, each containing a single point source
# first, make component list, then transfer to a model image
def makeComponentList(srclocindx):
    c = []
    # on-axis
    c.append(c_transit)
    # offset East at half-power radius (at freq)
    c.append(c_transit.directional_offset_by(90*u.deg,c0/(freq*1e9)/antdia/2*u.rad))
    
    srclocval = srcloc[srclocindx]
    slinfo    = 'on-axis'
    if srclocval == 1: slinfo = 'off-axis'
    clname = prefix+'_'+slinfo+'.cl'
    os.system('rm -rf '+clname)
    cl.done()
    fluxdensity = 1.0
    mydir  = 'J2000 '+c[srclocval].ra.to_string(u.hour)+' '+\
                      c[srclocval].dec.to_string(u.deg)
    cl.addcomponent(dir=mydir,flux=fluxdensity,fluxunit='Jy',freq=str(freq)+'GHz',
                    shape='point',spectrumtype='spectral index',index=spindx);
    cl.rename(filename=clname)
    cl.done()
    return c[srclocval],fluxdensity

def makeClnMask(srclocindx,cmploc,perrindx):
    # constuct mask for improved cleaning
    # circles about injected source locations with radius 2*PSF_FWHM
    srclocval = srcloc[srclocindx]
    slinfo    = 'on-axis'
    if srclocval == 1: slinfo = 'off-axis'
    clnmask = prefix+'_'+slinfo+'_perr-'+str(perr[perrindx])+'arcsec.crtf'
    radius  = 2*psf0*3600
    f = open(clnmask,'w')
    f.write('#CRTFv0\n')
    ra  = cmploc.ra.to_string(u.hour)
    dec = cmploc.dec.to_string(u.deg)
    f.write('circle[['+ra+', '+dec+'], '+str(radius)+'arcsec]\n')
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

# construct empty MS
def makeMSFrame(nchanindx,srclocindx,perrindx):
    numchan   = nchan[nchanindx]
    srclocval = srcloc[srclocindx]
    slinfo    = 'on-axis'
    if srclocval == 1: slinfo = 'off-axis'
    msname1 = prefix+'_'+slinfo+'_nchan-'+str(numchan)+'_perr-'+\
              str(perr[perrindx])+'asec_TEMP1.ms'
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
    sm.setfield(sourcename='pointing-1',sourcedirection=me.direction(rf='J2000',
                v0=c_transit.ra.to_string(u.hour),v1=c_transit.dec.to_string(u.deg)));
    sm.setlimits(shadowlimit=shadowlimit,elevationlimit=elevationlimit);
    sm.setauto(autocorrwt=0.0);
    # legacy from sim_pointing_SBA.py , retain here without any harm
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
    sm.observe(sourcename='pointing-1',spwname=spwname,
               starttime=str(-obsdur/2/60)+'h',stoptime=str(obsdur/2/60)+'h');
    sm.close();
    # flagdata(vis=msname1,mode='unflag')  # commented in case elevation/shadow flags matter

# transfer component list into model images for all pointing directions
# no need to sample this with cell_im or imsize_im because the PB drops off very slowly
# compared to cell_im (also, doing so would be computationally prohibitive)
def makeModelImages(nchanindx,srclocindx):
    numchan   = nchan[nchanindx]
    srclocval = srcloc[srclocindx]
    slinfo    = 'on-axis'
    if srclocval == 1: slinfo = 'off-axis'
    clname = prefix+'_'+slinfo+'.cl'
    imname = prefix+'_'+slinfo+'_nchan-'+str(numchan)+'.model.im'
    # first, create empty image, then populate
    ia.close()
    ia.fromshape(imname,[imsize_pb,imsize_pb,1,numchan],overwrite=True)
    cs = ia.coordsys()
    cs.setunits(['rad','rad','','Hz'])
    cs.setincrement([np.deg2rad(-cell_pb),np.deg2rad(cell_pb)],'direction')
    cs.setreferencevalue([c_transit.ra.to_string(u.rad),
                          c_transit.dec.to_string(u.rad)],type='direction')
    cs.setreferencevalue(str(freq)+'GHz','spectral')
    cs.setreferencepixel([(numchan-1)/2],'spectral')
    cs.setincrement(str(chanbw)+'MHz','spectral')
    ia.setcoordsys(cs.torecord())
    ia.setbrightnessunit('Jy/pixel')
    ia.set(0.0)
    ia.close()
    # populate model image with component list
    cl.open(clname)
    ia.open(imname)
    ia.modify(cl.torecord(),subtract=False)
    ia.close()
    cl.done()
    if testing: exportfits(imagename=imname,fitsimage=imname.replace('.im','.fits'))

# function to construct template image to later store antenna/baseline primary beams
# (legacy: and antenna aperture illumination functions)
# note: it isn't necessary to save PBs (legacy: or AIFs) in this code, but they
#       can be useful to enable easy inspection
def makeTemplateImg(nchanindx):
    numchan = nchan[nchanindx]
    imname  = prefix+'_nchan-'+str(numchan)+'.template-aif-pb.im'
    ia.close()
    ia.fromshape(imname,[imsize_pb,imsize_pb,1,numchan],overwrite=True)
    cs = ia.coordsys()
    cs.setunits(['rad','rad','','Hz'])
    cs.setincrement([np.deg2rad(-cell_pb),np.deg2rad(cell_pb)],'direction')
    cs.setreferencevalue([c_transit.ra.to_string(u.rad),
                          c_transit.dec.to_string(u.rad)],type='direction')
    cs.setreferencevalue(str(freq)+'GHz','spectral')
    cs.setreferencepixel([(numchan-1)/2],'spectral')
    cs.setincrement(str(chanbw)+'MHz','spectral')
    ia.setcoordsys(cs.torecord())
    ia.setbrightnessunit('Jy/pixel')
    ia.set(0.0)
    ia.close()
    return imname,cs.torecord()

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

def arrayFromImg(fname,c,fromfits):
    # arr will be [imsize,imsize,stokes=1,nchan]
    # if c=-1 then full cube will be returned, otherwise single channel
    if fromfits:
        importfits(fitsimage=fname.replace('.im','.fits'),
                   imagename=fname,defaultaxes=True,
                   defaultaxesvalues=['ra','dec','stokes','freq']);
    
    myia = image()
    myia.open(fname)
    arr  = myia.getchunk([-1,-1,0,c],[-1,-1,0,c])
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
    # fid not needed in this single-pointing code, inherited from SBA code, harmless
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
    # also return t
    return parang,t

def simVis(msname,srclocval,c,tstart,tend,c_transit,fd):
    # make temporary corrupted component list
    clname0 = msname.split('_')[0]+'_on-axis.cl'
    if srclocval == 1: clname0 = clname0.replace('on-axis','off-axis')
    clname = msname.replace('.ms','.cl')
    os.system('cp -r '+clname0+' '+clname)
    cl.open(clname,nomodify=False,log=False)
    cl.setflux(0,[fd,0,0,0],unit='Jy')
    cl.close(log=False)
    
    im.open(msname,usescratch=True)   # ft can only write to MODEL
    im.selectvis(baseline='00&&01',nchan=1,start=c,field=0,
                 time=[tstart+'~'+tend])
    # pass characteristics of phase center to im
    # nx,cy,cellx,celly are not utilized with complist ft
    #   so set them to arbitrary values
    im.defineimage(stokes='I',mode='channel',
                   nx=100,ny=100,cellx='1arcsec',celly='1arcsec',
                   phasecenter=me.direction(c_transit.frame.name,
                   v0=c_transit.ra.to_string(u.hour),
                   v1=c_transit.dec.to_string(u.deg)))
    # ft implements equivalent of gridder='standard' in tclean
    im.setoptions(ftmachine='ft')
    im.ft(complist=clname,incremental=False)
    im.done()
    os.system('rm -rf '+clname)

# legacy:
#def parallel_perAnt(t,c,pnterrors,parang,aif0c,testing,testingT,prefix,
#                    numchan,perr,perrindx,srclocval,templateImg,templateImgCoord,
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
#    slinfo = 'on-axis'
#    if srclocval == 1: slinfo = 'off-axis'
#    if (testing) and (t==testingT):
#        # phase-perturbed time/ant/freq-dependent pol-independent AIFs
#        # ignore the dummy coordinates that will come with these fits images
#        # the cells should really be in units of wavelength with
#        # pixel size = cell_uv and pixel range from -uvmax to +uvmax
#        # add extra dimensions to aif for export to image (npol,nchan)
#        outf = prefix+'_'+slinfo+'_nchan-'+str(numchan)+'_perr-'+\
#               str(perr[perrindx])+'_tstep-'+str(testingT)+'_ant-'+str(a+1)+\
#               '_phs-perturbed_aif-phs.im'
#        if c==0:
#            os.system('cp -r '+templateImg+' '+aifdir+'/'+outf)
#            temparr = np.zeros((imsize_pb,imsize_pb,1,numchan))
#        else:
#            temparr = arrayFromImg(aifdir+'/'+outf,-1,True)
#        
#        temparr2 = np.angle(aif,deg=True)
#        temparr2[aif==0] = 0
#        temparr[:,:,0,c] = temparr2
#        arrayToImg(temparr,aifdir+'/'+outf,templateImgCoord,True)
#        
#        # phase-perturbed time/ant/freq-dependent pol-independent PBs
#        outf = prefix+'_'+slinfo+'_nchan-'+str(numchan)+'_perr-'+\
#               str(perr[perrindx])+'_tstep-'+str(testingT)+\
#               '_ant-'+str(a+1)+'_phs-perturbed_pb.im'
#        if c==0:
#            os.system('cp -r '+templateImg+' '+pbAdir+'/'+outf)
#            temparr = np.zeros((imsize_pb,imsize_pb,1,numchan))
#        else:
#            temparr = arrayFromImg(pbAdir+'/'+outf,-1,True)
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
                    numchan,perr,perrindx,srclocval,templateImg,templateImgCoord,
                    imsize_pb,pbAdir,antdia,a,def_param=(volt,uvals0,vvals0)):
    casalog.filter('SEVERE')
    slinfo = 'on-axis'
    if srclocval == 1: slinfo = 'off-axis'
    
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
        outf = prefix+'_'+slinfo+'_nchan-'+str(numchan)+'_perr-'+\
               str(perr[perrindx])+'_tstep-'+str(testingT)+\
               '_ant-'+str(a+1)+'_phs-perturbed_pb.im'
        if c==0:
            os.system('cp -r '+templateImg+' '+pbAdir+'/'+outf)
            temparr = np.zeros((imsize_pb,imsize_pb,1,numchan))
        else:
            temparr = arrayFromImg(pbAdir+'/'+outf,-1,True)
        
        # square voltage amplitude to get power
        # result is already normalized to unity
        temparr[:,:,0,c] = (volt[:,:,a,0])**2
        arrayToImg(temparr,pbAdir+'/'+outf,templateImgCoord,True)

def parallel_perBl(t,c,pnterrors,parang,sigma,testing,testingT,
                   prefix,numchan,perr,perrindx,srclocval,templateImg,
                   templateImgCoord,pbBdir,imsize_pb,timestamp,tint,
                   c_transit,msname_bl,def_param=(volt,uvals0,vvals0)):
    casalog.filter('SEVERE')
    slinfo = 'on-axis'
    if srclocval == 1: slinfo = 'off-axis'
    
    ### construct sky-frame perturbed baseline-dependent primary
    ### beams at this time/freq (polarization-independent)
    
    a1 = int(re.split('[_ .]',msname_bl)[-3])
    a2 = int(re.split('[_ .]',msname_bl)[-2])
    
    if (c==0) and (t==0):
        # hack MS partition
        # many threads will attempt this at the same time
        # ok because read-only
        msname1 = msname_bl.replace('_'+str(a1)+'_'+str(a2)+'.ms','.ms')
        split(vis=msname1,outputvis=msname_bl,datacolumn='all',
              antenna=str(a1).zfill(2)+'&&'+str(a2).zfill(2),keepflags=False)
    
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
    #bpb   *= attnAmp / np.max(np.abs(bpb))   # legacy
    bpb    *= attnAmp / np.max(bpb)
    
    if (testing) and (t==testingT):
        # amp+phase-perturbed time/ant/freq-dependent
        # pol-independent baseline PBs
        outf = prefix+'_'+slinfo+'_nchan-'+str(numchan)+'_perr-'+\
               str(perr[perrindx])+'_tstep-'+str(testingT)+\
               '_ant1-'+str(a1+1)+'_ant2-'+str(a2+1)+\
               '_amp-phs-perturbed_bpb.im'
        if c==0:
            os.system('cp -r '+templateImg+' '+pbBdir+'/'+outf)
            temparr = np.zeros((imsize_pb,imsize_pb,1,numchan))
        else:
            temparr = arrayFromImg(pbBdir+'/'+outf,-1,True)
        
        #temparr[:,:,0,c] = np.abs(bpb)        # legacy
        temparr[:,:,0,c] = bpb
        arrayToImg(temparr,pbBdir+'/'+outf,templateImgCoord,True)
        del temparr
    
    ### construct true sky image seen by this baseline
    ### (polarization-independent)
    
    temparr = arrayFromImg(prefix+'_'+slinfo+'_nchan-'+str(numchan)+\
                           '.model.im',c,False)
    #temparr[:,:,0,0] *= np.abs(bpb)           # legacy
    temparr[:,:,0,0]  *= bpb
    
    # for an input point source model, temparr will contain a single
    # pixel with non-zero flux density.  This is the corrupted
    # flux density of interest.
    fd = np.max(temparr)
    del temparr
    
    ### simulate visibilities from the corrupted sky model into
    ### the MODEL column for all integrations in this timestep
    
    tstart = qa.time(qa.quantity(timestamp-tint/2,unitname='s'),form='ymd')[0]
    tend   = qa.time(qa.quantity(timestamp+tint/2,unitname='s'),form='ymd')[0]
    # simVis is a time sink, which explains the gymnastics
    # with the parallelization and MSname disk space usage
    # (don't bother using the MS partition route, a-la pieflag)
    simVis(msname_bl,srclocval,c,tstart,tend,c_transit,fd)

# corrupt visibilities
def corruptVis(nchanindx,srclocindx,perrindx,templateImg,templateImgCoord,pool):
    numchan   = nchan[nchanindx]
    srclocval = srcloc[srclocindx]
    slinfo    = 'on-axis'
    if srclocval == 1: slinfo = 'off-axis'
    msname    = prefix+'_'+slinfo+'_nchan-'+str(numchan)+'_perr-'+\
                str(perr[perrindx])+'asec.ms'     # split msname1 here (data col)
    msname1   = msname.replace('.ms','_TEMP1.ms') # template, concat bl's here later
    
    # calculate parallactic angle per ant/integration, to enable later translation
    # of pointing errors from X-Y to U-V coordinate frames
    # note: src decl < obs lat here, so parang is negative as src rises in East
    # parang [time,ant] in radians
    # also returns timestamps (all integrations, UTC seconds)
    parang,timestamps = calcParang(msname1)
    
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
        #outf = prefix+'_'+slinfo+'_nchan-'+str(numchan)+'_perr-'+\
        #       str(perr[perrindx])+'_unperturbed_aif-abs.im'
        #os.system('cp -r '+templateImg+' '+aifdir+'/'+outf)
        ## ignore the dummy coordinates that will come with these fits images
        ## the cells should really be in units of wavelength with
        ## pixel size = cell_uv and pixel range from -uvmax to +uvmax
        ## add extra dimension to aif0 for export to image
        #temparr = np.zeros((imsize_pb,imsize_pb,1,numchan),'complex')
        #temparr[:,:,0,:] = aif0
        #arrayToImg(np.abs(temparr),aifdir+'/'+outf,templateImgCoord,True)
        
        # unperturbed freq-dependent ant/pol/time-independent PBs
        outf = prefix+'_'+slinfo+'_nchan-'+str(numchan)+'_perr-'+\
               str(perr[perrindx])+'_unperturbed_pb.im'
        os.system('cp -r '+templateImg+' '+pbAdir+'/'+outf)
        temparr = np.zeros((imsize_pb,imsize_pb,1,numchan))
        for c in range(numchan): temparr[:,:,0,c] = (makeVolt(lam[c],antdia,0,0))**2
        
        arrayToImg(temparr,pbAdir+'/'+outf,templateImgCoord,True)
        del temparr
    
    # assign pointing errors as complex numbers for all antennas in all integrations
    # +real=+X +imag=+Y; phase 0=X, increases North through East
    # pointing error = radial offset (amp), in radians
    myscale   = np.deg2rad(perr[perrindx]/3600)/np.sqrt(np.pi/2) # rayleigh statistics
    pnterrors = np.random.normal(size=(ntsteps,nant),scale=myscale) +\
                np.random.normal(size=(ntsteps,nant),scale=myscale)*1j
    # save full output in case results from random instance needed later (unlikely?)
    outf = prefix+'_'+slinfo+'_nchan-'+str(numchan)+'_perr-'+\
           str(perr[perrindx])+'asec_pnterrors.pickle'
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
                # MS partition takes place in parallel_perBl to prevent holdup here
    
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
            
            f = partial(parallel_perAnt,t,c,lam[c],pnterrors,parang,testing,testingT,
                        prefix,numchan,perr,perrindx,srclocval,templateImg,
                        templateImgCoord,imsize_pb,pbAdir,antdia)
            pool.map(f,range(nant))
            
            ### corrupt visibilities per baseline
            
            f = partial(parallel_perBl,t,c,pnterrors,parang,sigma[c],
                        testing,testingT,prefix,numchan,perr,perrindx,
                        srclocval,templateImg,templateImgCoord,pbBdir,
                        imsize_pb,timestamps[t],tint,c_transit)
            pool.map(f,bl_list)
    
    # concatenate serially to prevent corruption
    for a1 in range(nant):
        for a2 in range(nant):
            if a1 < a2:
                msname_bl = msname1.replace('.ms','_'+str(a1)+'_'+str(a2)+'.ms')
                ms.open(msname_bl)
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
def makeImg(nchanindx,srclocindx,perrindx,vptab,cmploc):
    print (' imaging with clean fractional threshold = '+str(cleanfrac[perrindx]))
    numchan   = nchan[nchanindx]
    srclocval = srcloc[srclocindx]
    slinfo    = 'on-axis'
    if srclocval == 1: slinfo = 'off-axis'
    pc        = 'J2000 '+cmploc.ra.to_string(u.hour)+' '+cmploc.dec.to_string(u.deg)
    clnmask   = prefix+'_'+slinfo+'_perr-'+str(perr[perrindx])+'arcsec.crtf'
    msname    = prefix+'_'+slinfo+'_nchan-'+str(numchan)+'_perr-'+\
                str(perr[perrindx])+'asec.ms'
    imgdir    = msname.replace('.ms','')
    os.system('rm -rf '+imgdir)
    os.mkdir(imgdir)
    os.chdir(imgdir)
    if numchan == 1:
        # dirty image
        tclean(vis='../'+msname,imagename='sim',imsize=imsize_im,phasecenter=pc,
               cell='{:f}'.format(cell_im)+'arcsec',gridder='standard',
               vptable='../'+vptab,weighting='natural',niter=0);
        mystats = imstat(imagename='sim.residual',listit=False,verbose=False);
        mythreshold = mystats['max'][0] * cleanfrac[perrindx]
        #exportfits(imagename='sim.image',fitsimage='sim.image.dirty.fits')
        #exportfits(imagename='sim.psf',fitsimage='sim.psf.fits')
        casalog.filter('INFO')   # reinstate info for imaging (after dirty image)
        # cleaned image
        tclean(vis='../'+msname,imagename='sim',imsize=imsize_im,phasecenter=pc,
               cell='{:f}'.format(cell_im)+'arcsec',gridder='standard',
               vptable='../'+vptab,weighting='natural',niter=1000,mask=['../'+clnmask],
               threshold=mythreshold,calcres=False,calcpsf=False);
        casalog.filter('SEVERE')
        #exportfits(imagename='sim.image',fitsimage='sim.image.fits')
        #exportfits(imagename='sim.residual',fitsimage='sim.residual.fits')
        #exportfits(imagename='sim.model',fitsimage='sim.model.fits')
        #exportfits(imagename='sim.pb',fitsimage='sim.pb.fits')
        casalog.filter('INFO')
        # PB-corrected image
        tclean(vis='../'+msname,imagename='sim',imsize=imsize_im,phasecenter=pc,
               cell='{:f}'.format(cell_im)+'arcsec',gridder='standard',
               vptable='../'+vptab,weighting='natural',niter=0,mask=['../'+clnmask],
               calcres=False,calcpsf=False,pbcor=True);
        casalog.filter('SEVERE')
        #exportfits(imagename='sim.image.pbcor',fitsimage='sim.image.pbcor.fits')
    else:
        # dirty image
        tclean(vis='../'+msname,imagename='sim',imsize=imsize_im,phasecenter=pc,
               cell='{:f}'.format(cell_im)+'arcsec',gridder='standard',
               vptable='../'+vptab,weighting='natural',deconvolver='mtmfs',
               nterms=nterms,niter=0);
        mystats = imstat(imagename='sim.residual.tt0',listit=False,verbose=False);
        mythreshold = mystats['max'][0] * cleanfrac[perrindx]
        #exportfits(imagename='sim.image.tt0',fitsimage='sim.image.tt0.dirty.fits')
        #exportfits(imagename='sim.psf.tt0',fitsimage='sim.psf.tt0.fits')
        casalog.filter('INFO')   # reinstate info for imaging (after dirty image)
        # cleaned image
        tclean(vis='../'+msname,imagename='sim',imsize=imsize_im,phasecenter=pc,
               cell='{:f}'.format(cell_im)+'arcsec',gridder='standard',
               vptable='../'+vptab,weighting='natural',deconvolver='mtmfs',
               nterms=nterms,niter=1000,mask=['../'+clnmask],threshold=mythreshold,
               calcres=False,calcpsf=False);
        casalog.filter('SEVERE')
        #exportfits(imagename='sim.image.tt0',fitsimage='sim.image.tt0.fits')
        #exportfits(imagename='sim.residual.tt0',fitsimage='sim.residual.tt0.fits')
        #exportfits(imagename='sim.model.tt0',fitsimage='sim.model.tt0.fits')
        #exportfits(imagename='sim.pb.tt0',fitsimage='sim.pb.tt0.fits')
        # PB-corrected image
        # don't investigate spectral index recovery in this code
        widebandpbcor(vis='../'+msname,imagename='sim',nterms=nterms,
                      action='pbcor',spwlist=[0]*numchan,
                      chanlist=list(range(numchan)),weightlist=[1]*numchan);
        #exportfits(imagename='sim.pbcor.image.tt0',
        #           fitsimage='sim.pbcor.image.tt0.fits')
    
    os.chdir('../')

# capture image dynamic range and fidelity statistics
def calcStats(nchanindx,srclocindx,perrindx,fdsum):
    print (' recording statistics')
    numchan   = nchan[nchanindx]
    srclocval = srcloc[srclocindx]
    slinfo    = 'on-axis'
    if srclocval == 1: slinfo = 'off-axis'
    imgdir    = prefix+'_'+slinfo+'_nchan-'+str(numchan)+'_perr-'+\
                str(perr[perrindx])+'asec'
    if numchan == 1:
        ia.open(imgdir+'/sim.image')
        img_max = ia.statistics()['max'][0]
        ia.close()
        ia.open(imgdir+'/sim.residual')
        res_rms = ia.statistics()['rms'][0]
        ia.close()
        ia.open(imgdir+'/sim.image.pbcor')
        imgPB_max = ia.statistics()['max'][0]
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
        ia.open(imgdir+'/sim.pbcor.image.tt0')
        imgPB_max = ia.statistics()['max'][0]
        ia.close()
    
    dr  = img_max / res_rms
    imf = 1 - np.abs(imgPB_max - fdsum) / fdsum
    return (dr,imf)

def main():
    if not imgOnly: pool  = multiprocessing.Pool(cpuCount)
    casalog.filter('SEVERE')     # suppress screen warnings (imaging excepted)
    vptab = makevptab()
    for k in range(len(srcloc)):      # number of source locations
        cmploc,fdsum = makeComponentList(k)
        for i in range(len(nchan)):           # number of channels
            if not imgOnly:
                makeModelImages(i,k)
                if k == 0:
                    templateImg,templateImgCoord = makeTemplateImg(i)
                    if not testing:
                        # If testing, keep template image to later copy and store AIFs
                        # and ant/bl PBs.  Store templateImgCoord whether testing or not
                        os.system('rm -rf '+templateImg)
            
            slinfo = 'on-axis'
            if srcloc[k] == 1: slinfo = 'off-axis'
            outf = prefix+'_nchan-'+str(nchan[i])+'_'+slinfo+'_results.txt'
            f = open(outf,'w')
            if nchan[i] > 1: f.write('# injected spectral index = '+str(spindx)+'\n')
            f.write('# perr (arcsec)    dynamic range    fidelity\n')
            for m in range(len(perr)):    # number of pointing errors
                print ('=== Processing '+slinfo+', nchan='+str(nchan[i])+\
                       ', pointing error='+str(perr[m])+'" ===')
                if not imgOnly:
                    if i==0: makeClnMask(k,cmploc,m)
                    makeMSFrame(i,k,m)
                    corruptVis(i,k,m,templateImg,templateImgCoord,pool)
                
                makeImg(i,k,m,vptab,cmploc)
                # record dynamic range and fidelity statistics
                dr,imf = calcStats(i,k,m,fdsum)
                f.write('%10.1f %19.2e %13.2e\n' % (perr[m],dr,imf))
            
            f.close()
    
    if not imgOnly:
        pool.close()
        pool.join()

if __name__ == '__main__':
    main()
