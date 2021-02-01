import numpy as np
import glob
import os
import sys
import re
import subprocess
import fitsio
import healpy as hp
import warnings

from astropy.io import fits
from astropy.table import Table, Column
import astropy.coordinates as coords
import astropy.units as u

import matplotlib.pyplot as pl

from desitarget.sv1.sv1_targetmask import desi_mask, mws_mask, bgs_mask

import desi_retriever
import desi_retriever.andes

############################################################

class MWSData():
    
    MWS_REDUX_DIR = '/global/cfs/cdirs/desi/science/mws/redux/'
    
    def __init__(self,redux='blanc',run=210112):
        """
        """
        self.redux_dirname = redux
        
        # Special cases
        if redux == 'sv_daily':
            self.redux = 'daily'
        else:
            self.redux = redux
        
        self.run = str(run)
        self.load_tables()   
        return
    
    def __repr__(self):
        return '{}-{}'.format(redux,coadd)
    
    def __getitem__(self,key):
        return self.fm[key]
    
    def rvtab_path(self):
        """
        """
        base = 'rvtab_spectra-{}.fits'.format(self.redux)
        return os.path.join(self.MWS_REDUX_DIR,self.redux_dirname,'rv_output',self.run,base)
    
    def load_tables(self):
        # Load tables
        self.fm = Table.read(self.mwtab_path(),'FIBERMAP')   
        try:
            self.rv = Table.read(self.mwtab_path(),'SPRVTAB ')    
        except KeyError:
            self.rv = Table.read(self.mwtab_path(),'RVTAB ')   
            
############################################################

class MWSExposures(MWSData):
    """
    Single exposures.
    
    Example:
        d = MWSExposures(redux='sv_daily', run=210110)
    """
    def __repr__(self):
        return 'MWS: {}-exp-{} ({})'.format(self.redux, self.run, self.mwtab_path())
    
    def mwtab_path(self):
        """
        For single exposures there is no combined table, yet
        """
        return self.rvtab_path()
    
    def fetch_spec_for_targetid(self,targetid,iexp=0,with_model=True):
        """
        """
        i = np.flatnonzero(self['TARGETID'] == targetid)[iexp]
        
        tileid = self.fm['TILEID'][i]
        night = self.fm['NIGHT'][i]
        fiber = self.fm['FIBER'][i]
            
        D = desi_retriever.andes.fetcher.get_specs(tileid=tileid,
                                               night=night,
                                               fiber=fiber,
                                               targetid=targetid,
                                               coadd=False,
                                               dataset=self.redux)
        
        if with_model:
            Dmod = desi_retriever.andes.fetcher.get_rvspec_models(tileid=tileid,
                                                        night=night,
                                                        fiber=fiber,
                                                        targetid=targetid, 
                                                        run=self.run,
                                                        coadd=False,
                                                        dataset=self.redux_dirname)
                
        if with_model:
            return D, Dmod
        else:
            return D
        
############################################################
        
class MWSCoadd(MWSData):
    """
    Coadds are by tile, either:
        - nightly
        - all
        - deep
    
    Not all tiles have all coadd types.
    
    Example:
         # Read blanc coadds for all exposures
         MWSCoadd(coadd='all') 
    """
    def __init__(self,redux='blanc',coadd='all'):
        assert(coadd is not None)
        self.coadd = coadd
        super().__init__(redux)
            
    def __repr__(self):
        return 'MWS: {}-{} ({})'.format(self.redux,self.coadd, self.mwtab_path())
  
    def mwtab_path(self):
        """
        For the coadds, there is a 'master' table.
        """
        base = 'mwtab_coadd-{}-{}.fits'.format(self.redux,self.coadd)
        return os.path.join(self.MWS_REDUX_DIR,self.redux,base)
    
    def fetch_spec_for_targetid(self,targetid,night=None,with_model=True):
        """
        """
        i = np.flatnonzero(self['TARGETID'] == targetid)
        
        tileid = self.fm['TILEID'][i]
        fiber = self.fm['FIBER'][i]
        
        if self.coadd == 'nights':
            assert(night is not None)
            
        D = desi_retriever.andes.fetcher.get_specs(tileid=tileid,
                                               night=night,
                                               fiber=fiber,
                                               targetid=targetid,
                                               coadd=True,
                                               dataset=self.redux)
        
        return D
############################################################

def plot_targetid(d,targetid,iexp=0):
    """
    """
    pl.figure(figsize=(15,8))
    D, Dmod = d.fetch_spec_for_targetid(targetid,iexp=iexp, with_model=True)

    ispec = 0
    ax0 = pl.subplot(411,label='b')
    pl.plot(D[ispec]['b_wavelength'],D[ispec]['b_flux'],color='lightblue');

    ax1 = pl.subplot(412,label='r')
    pl.plot(D[ispec]['r_wavelength'],D[ispec]['r_flux'],color='lightgreen');

    ax2 = pl.subplot(413,label='z')
    pl.plot(D[ispec]['z_wavelength'],D[ispec]['z_flux'],color='red');

    ax3 = pl.subplot(414,label='all')
    pl.plot(D[ispec]['b_wavelength'],D[ispec]['b_flux'],color='k')
    pl.plot(D[ispec]['r_wavelength'],D[ispec]['r_flux'],color='k')
    pl.plot(D[ispec]['z_wavelength'],D[ispec]['z_flux'],color='k') 
    
    pl.sca(ax0);pl.plot(Dmod[ispec]['b_wavelength'],Dmod[ispec]['b_model'],color='blue');
    pl.sca(ax1);pl.plot(Dmod[ispec]['r_wavelength'],Dmod[ispec]['r_model'],color='g');
    pl.sca(ax2);pl.plot(Dmod[ispec]['z_wavelength'],Dmod[ispec]['z_model'],color='purple')
    
    for ax,band in zip([ax0,ax1,ax2,ax3],['b','r','z','r']):
        pl.sca(ax)
        fluxname = '{}_flux'.format(band)
        positive_flux = D[ispec][fluxname] > 0
        ymin = np.maximum(-10,np.min(D[ispec][fluxname]))
        ymax = 3*np.median(D[ispec][fluxname][positive_flux])
        pl.ylim(ymin,ymax)

    mag_g = 22.5-2.5*np.log10(d['FLUX_G'][iexp])
    mag_r = 22.5-2.5*np.log10(d['FLUX_R'][iexp])  
    
    return D,Dmod

    
############################################################

def healpix_from_table(t,nside=8):
    """
    """
    theta, phi = np.radians(90-t['TARGET_DEC']), np.radians(t['TARGET_RA'])
    return hp.ang2pix(nside,theta,phi,nest=True)

############################################################

def get_images(t,zoom=14):
    """
    """
    from IPython.core.display import display, HTML
    tag = '<img src="http://legacysurvey.org//viewer/jpeg-cutout?ra={ra:f}&dec={dec:f}&zoom={zoom:}&layer=ls-dr9">'
    tags = '\n'.join([tag.format(ra=_['TARGET_RA'],dec=_['TARGET_DEC'],zoom=zoom) for _ in t])
    display(HTML(tags))

############################################################

def target_healpix_path(ipix,resolve='bright'):
    """
    """
    TARGETS_DIR = '/global/cfs/cdirs/desi/target/catalogs/dr9/0.47.0/targets/sv1/resolve/'
    return os.path.join(TARGETS_DIR,resolve,'sv1targets-bright-hp-{}.fits'.format(ipix))