import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import stpsf
import os
import sep
import glob

from astropy.io import fits     
from astropy.convolution import convolve_fft
from joblib import Parallel, delayed
from pathlib import Path
from scipy.optimize import curve_fit
from astropy.stats import sigma_clipped_stats, sigma_clip
from lacosmic.core import lacosmic # func is apparently deprecated - will be 'remove_cosmics'
from sklearn.cluster import DBSCAN

import warnings
warnings.filterwarnings("ignore")

import logging, sys
logging.disable(sys.maxsize)

np.set_printoptions(legacy='1.25') # my environment is a bit wonky :/

def linear_fitting(coords,cube,n_int,n_group):
    """
    fits ramps piecewise with 2 straight lines if a jump is detected,
    otherwise fits a single line. If a piece is too short, fits only the longer one.
    """
    row, col = coords

    grads = [] # straight line gradient based on first data points
    intercepts = [] # straight line intercept based on first data points
    resids = [] # residuals from curve_fit of power law

    for int_num in range(n_int):
        x_dat = [i + (int_num)*n_group for i in range(n_group)]
        y_dat = [cube[x][row][col] for x in x_dat]
        x = np.asarray(x_dat, dtype=float) 
        y = np.asarray(y_dat, dtype=float)

        # removing first and last frames
        y[0] = np.nan
        y[-1] = np.nan

        mask = ~np.isnan(y)
        x = x[mask]
        y = y[mask]

        p,r,*_ = np.polyfit(x,y,1,full=True)
        m = p[0]
        c = p[1]
        grads.append(m)
        intercepts.append(c)
        resids.append(r[0] if len(r) > 0 else np.nan)
        
    return [row, col, grads, intercepts, resids]


def run_lacosmic(frame_data, mask):
    """
    running lacosmic so it can be done parallely on frames
    mask: boolean mask where True = science pixel, False = bad pixel
    """
    from astropy import log
    log.setLevel('ERROR') 

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    masked_arr = np.where(mask, frame_data, np.nan)
    _, _, std = sigma_clipped_stats(masked_arr)
    error_arr = np.full(frame_data.shape, std)
    lacosmic_mask = ~mask # (True = masked/bad pixel)
    data_clean = np.nan_to_num(frame_data, nan=0.0, posinf=0.0, neginf=0.0) # replace nans which lacosmic doesn't like

    clean, crmask = lacosmic(data_clean,contrast=4,cr_threshold=2,
                             neighbor_threshold=0.9,mask=lacosmic_mask,error=error_arr)

    return clean, crmask

def _run_sep(frame, data, kernel, mask, save, obs_dir,n_group):
    """
    run source extractor in parallel
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)

    data = np.array(data, dtype=np.float32, copy=True)
    mask = np.array(mask, dtype=bool, copy=True)
    # get and subtract the background
    bkg = sep.Background(data, mask=~mask)
    data_sub = data - bkg

    # object detection
    objects = sep.extract(data_sub, 1.5, filter_kernel=kernel, err=bkg.globalrms, mask=~mask)
    obj_df = pd.DataFrame(objects)

    if len(obj_df) == 0:
        # return empty filtered_df with same cols
        filtered_df = pd.DataFrame(columns=obj_df.columns.tolist() + ['symmetry', 'frame', 'sep_flux', 'sep_fluxerr','sep_s/n', 'sep_flag'])
        return obj_df, filtered_df

    # adding needed cols
    obj_df['symmetry'] = (obj_df['a']/obj_df['b']).abs() - 1
    obj_df['frame'] = frame
    obj_df['group'] = (obj_df['frame'] % n_group) + 1 # adding the group number

    # aperture photometry
    flux, fluxerr, flag = sep.sum_circle(data_sub, obj_df['x'], obj_df['y'], 3.0, err=bkg.globalrms, gain=1.0) # ap radius = 3.0
    obj_df['sep_flux'] = flux
    obj_df['sep_fluxerr'] = fluxerr
    obj_df['sep_s/n'] = flux/fluxerr
    obj_df['sep_flag'] = flag

    # apply filtering
    filter_mask = ((obj_df['symmetry'] < 0.5) & # used to be 1.5
                    obj_df['x'].between(16, 1016) &
                    obj_df['y'].between(12, 1012))
    filtered_df = obj_df[filter_mask]

    # plotting
    if save and len(filtered_df) > 0:
        from matplotlib.patches import Ellipse
        matplotlib.use("Agg") # don't show em
        fig, ax = plt.subplots()
        m, s = np.mean(data_sub*mask), np.std(data_sub*mask)

        im = ax.imshow(data_sub*mask,interpolation='nearest',vmin=m-s,vmax=m+s,origin='lower')
        ax.set_title(f"Frame {frame}")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Pixel value")

        for i in filtered_df.index:
            e = Ellipse(
                xy=(filtered_df.at[i, 'x'], filtered_df.at[i, 'y']),
                width=6*filtered_df.at[i, 'a'],
                height=6*filtered_df.at[i, 'b'],
                angle=filtered_df.at[i, 'theta']*180./np.pi
            )
            e.set_facecolor('none')
            e.set_edgecolor('red')
            ax.add_artist(e)

        sep_dir = os.path.join(obs_dir, "sep_frames")
        os.makedirs(sep_dir, exist_ok=True)
        plt.savefig(os.path.join(sep_dir, f"frame_{frame:03d}.png"), bbox_inches="tight")
        plt.close(fig)

    return obj_df, filtered_df


def make_reference_cube(pixel,grad_cube):
    """
    makes a reference cube and gets a median from it
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)

    row,col = pixel
    pix = grad_cube[:,row,col]

    clipped_pix = sigma_clip(pix,sigma_upper=1,masked=False,axis=0).data

    return np.asarray(clipped_pix, dtype=float)


class Jurassic():
    """
        Class for searching the ramps of full array MIRI images for fast transients

        JURASSIC: JWST Up the Ramp Analysis Searching the Sky for Infrared Transients
    """

    def __init__(self,file=None,num_cores=35,run=True,method='mega',ramps=None,images=True,
                 significance=True,mask_correction=True,plot=True,no_sat_mask=False,
                 base_dir=None,data_dir=None):
        """
        Initialise or whatevs

        Parameters
        ----------
        file : str
                File name of the observation

        method : str
                either 'ramp' or 'mega'
        

        other stuff I guess - will update at some point
        """
        self.file = file
        parts = self.file.replace('\\', '/').split('/')
        self.name = parts[-2] if len(parts) >= 2 else '.'
        self.obs_id = parts[-1]
        self.method = method
        self.plot = plot
        self.base_dir = base_dir
        self.data_dir = data_dir
        self.mask_correction = mask_correction
        self.no_sat_mask = no_sat_mask
        self.num_cores = num_cores # number of cores to use when running functions (sep, 1st order polyfit, lacosmic) in parallel
        self.psf_fwhm_px = { # taken from JDOX
            "F560W": 1.882,
            "F770W": 2.445,
            "F1000W": 2.982,
            "F1130W": 3.409,
            "F1280W": 3.818,
            "F1500W": 4.436,
            "F1800W": 5.373,
            "F2100W": 6.127,
            "F2550W": 7.300,
        }
        
        if run:
            self._assign_data()
            self.ramp_correction(cube=self.data)
            self.flux_calibrate(cube=self.data_cor)
            self._make_cubes()
            self._mask_pixels()

            if ramps: # search on ramp level
                print('ramps')
                self.parallel_fit_df(self.rampy_cube) # only fitting rampy_cube not mega

            if self.method == 'mega':
                if images or significance:
                    print('images')
                    self.mega_inator(self.rampy_cube)          # use raw rampy_cube
                    self._cube_gradient(self.mega_cube_masked, save=True)
                    self._reference_frame()
                    self._cube_differenced(self.grad_cube, self.first_ref_frame, save=False, first=None)
                    self._psf_kernel()
                    self.source_extracting(self.diff_cube, save_plot=False, save_csv=True)

                    print('re-difference')
                    self._masked_reference(self.mask_correction)
                    self._cube_differenced(self.grad_cube, self.second_ref_frame, save=True)
                    self._remove_cosmic(self.diff_cube)
                    self._make_ref_cr_mask()
                    self.source_extracting(self.clean_cube, save_plot=True, save_csv=False)

                if significance:
                    print('significance')
                    self._cube_significance()
                    self._cube_threshold() 
                    self._cube_rolling_sum()
                    self._significance_output()
            
            if self.method == 'ramp':
                if images or significance: # search on image level
                    print('images')
                    self._cube_gradient(self.rampy_cube,save=True)
                    self._reference_frame()
                    self._cube_differenced(self.grad_cube,self.first_ref_frame,save=False,first=None) # first saves first iteration of difference cube  
                    self._psf_kernel()
                    self.source_extracting(self.diff_cube,save_plot=False,save_csv=False)

                    print('re-difference')
                    self._masked_reference(self.mask_correction) # creating new mask and doing differencing again
                    self._cube_differenced(self.grad_cube,self.second_ref_frame,save=True)
                    self._remove_cosmic(self.diff_cube)
                    self._make_ref_cr_mask()
                    self.source_extracting(self.clean_cube,save_plot=True,save_csv=False)

                if significance:
                    print('significance')
                    self._cube_significance()
                    self._cube_threshold() 
                    self._cube_rolling_sum()
                    self._significance_output()

            self._time_mjd()
            self.save_outputs()

            if not hasattr(self, 'significance_df'):
                self.significance_df = pd.DataFrame()


    def _assign_data(self):
        """
        Opens the fits file and assigns the data to the class
        """
        # base outputs folder — defaults to outputs/ relative to cwd
        if self.base_dir is None:
            self.base_dir = os.path.join(os.getcwd(), 'outputs')
        os.makedirs(self.base_dir, exist_ok=True)

        # data folder — defaults to the directory containing the ramp file
        if self.data_dir is None:
            self.data_dir = os.path.dirname(os.path.abspath(self.file))

        # Remove the filename suffix
        suffix1 = '_mirimage_ramp.fits'
        suffix2 = '_mirimage_cal.fits'
        if self.obs_id.endswith(suffix1):
            obs_n = self.obs_id[:-len(suffix1)]
        else:
            obs_n = self.obs_id  # fallback

        obs_name = f"dr_{obs_n}" 

        # directory for specific observation/segment
        self.obs_dir = os.path.join(self.base_dir, obs_name)
        os.makedirs(self.obs_dir, exist_ok=True)

        # get level 2a (ramp.fits) data
        self.stage1_filepath = os.path.abspath(self.file)
        if not os.path.exists(self.stage1_filepath):
            print(f"Cannot find the Stage 1 file: {self.stage1_filepath}")

        # get level 2b (cal.fits) data
        try:
            self.stage2_filepath = os.path.join(self.data_dir, obs_n + suffix2)
            self.do_flux_cal = os.path.exists(self.stage2_filepath)
            if not self.do_flux_cal:
                print("Cannot find the Stage 2 file --- No flux calibration will be performed")
        except OSError:
            print("Cannot find the Stage 2 file --- No flux calibration will be performed")
            self.do_flux_cal = False

        # assigning data from ramp.fits file
        with fits.open(self.stage1_filepath) as hdul:
            self.data = hdul[1].data # science data
            self.dq_2d_arr = hdul[2].data # data quality flag array for whole cube
            self.dq_3d_arr = hdul[3].data # data quality flag array for each group
            times = hdul[7].data
            self.time_df = pd.DataFrame(times)
            self.tgroup = hdul['PRIMARY'].header['TGROUP']
            self.filename = hdul[0].header['FILENAME']
            self.filter = hdul[0].header['FILTER']
            self.subarray = hdul[0].header['SUBARRAY']
            self.targname = hdul[0].header['TARGNAME']

        self.n_int = len(self.data) # number of integrations (ramps) in file
        self.n_group = len(self.data[0]) # number of groups per integration
        self.n_frame = self.n_int * self.n_group # number of frames in file
        self.frames = list(range(self.n_frame)) # list of all frame indices

        bad_frames = []
        for integration in list(range(self.n_int)):
            bad_frames.append(integration*self.n_group)
            bad_frames.append(((integration+1)*self.n_group)-1)
        self.bad_frames = bad_frames

        if self.filter in self.psf_fwhm_px:
            self.fwhm = self.psf_fwhm_px[self.filter]
        else:
            raise ValueError(f'Unknown filter {self.filter}: no FWHM available')

    
    def ramp_correction(self,cube):
        """
        Uses rampdoctor to correct both the brighter-fatter effect
        and the reset switch charge decay effects
        """
        from rampdoctor import RampDoctor
        self.gen_mask = np.load('full_MIRI_mask.npy') # for full array
        rd = RampDoctor(cube=cube,bg_mask=self.gen_mask,verbose=True)

        self.data_cor = rd.correct(diagnostics=True) 


    def flux_calibrate(self,cube):
        """
        Calibrates the 4-dimensional ramp data (self.data)
        Using the information from the reference files
        with the data model: MirImgPhotomModel
        """
        # first get the conversion factor from the reference files
        from jwst import datamodels
        from stpipe import crds_client

        with datamodels.open(self.stage1_filepath) as model:
            crds_params = model.get_crds_parameters()
            filt = model.meta.instrument.filter

        # get the photom reference file from CRDS that corresponds to this exposure
        photom_file = crds_client.get_reference_file(crds_params, 'photom', 'jwst')
        print(f"Using PHOTOM ref: {photom_file}")

        # open file and get information that matches the filter
        with datamodels.MirImgPhotomModel(photom_file) as phot:
            table = phot.phot_table
            mask = table['filter'] == filt
            row = table[mask]
            photmjsr = float(row['photmjsr'][0])
            uncertainty = float(row['uncertainty'][0])
            print(f"{self.stage1_filepath}  filter={filt}  PHOTMJSR={photmjsr:.4f} MJy/sr per DN/s  +/- {uncertainty:.4f}")
        self.flux_conv = photmjsr
        self.flux_uncert = uncertainty

        # apply the conversion to science data - first need to change from DN/group to DN/s
        group_times = np.arange(1,self.n_group+1) * self.tgroup
        data_rate = cube / group_times[np.newaxis,:,np.newaxis,np.newaxis] # DN/s

        self.flux_data = data_rate * self.flux_conv # MJy/sr


    def _make_cubes(self):
        """
        makes cube from 4d uncal file, also jump detected cube
        """
        # DN cube (for saturation masking)
        ramps_dn = np.array_split(self.data_cor, self.n_int, axis=0)
        self.rampy_cube_dn = np.squeeze(np.concatenate(ramps_dn, axis=1))

        # MJy/sr cube (for science)
        ramps_flux = np.array_split(self.flux_data, self.n_int, axis=0)
        self.rampy_cube = np.squeeze(np.concatenate(ramps_flux, axis=1))

        # make reference cube for jumps detected with calwebb_detector1
        dq_ints = np.array_split(self.dq_3d_arr,len(self.dq_3d_arr),axis=0)
        dq_cube = np.concatenate(dq_ints,axis=1)
        self.dq_cube = np.squeeze(dq_cube) # bitwise cube with all the dq flags

        flag = 4 # jump detected flag
        jump_cube = np.full(self.dq_cube.shape, False, dtype=bool)

        for frame in range(len(self.dq_cube)):
            jump_arr = (self.dq_cube[frame] & flag) == flag
            jump_cube[frame] = jump_arr

        self.jump_cube = jump_cube # a boolean cube where True is for jumps detected
        

    def _circle_app(self,rad):
        """
        Makes a kinda circular aperture, probably not worth using. - from ryan
        """
        mask = np.zeros((int(rad*2+.5)+1, int(rad*2+.5)+1))
        c = rad
        x,y = np.where(mask==0)
        dist = np.sqrt((x-c)**2 + (y-c)**2)

        ind = (dist) < rad + .2
        mask[y[ind],x[ind]] = 1

        return mask
    

    def _mask_pixels(self,threshold = 45000): # could be udated w/ quality flags from JWST
        """
        returns a list of tuples that are pixel (row,col) coordinates
        that have masked out the non-science and saturated pixels
        threshold used to be 47000 but that let things pass through that we didn't want
        """
        # load general miri mask
        mask = self.gen_mask

        # mask out pixels that get counts above threshold
        mask_sat = self.rampy_cube_dn[-1] < threshold
        mask_sat = mask_sat.astype(int) # to convolve with aperture

        kernel = self._circle_app(10)

        mask_sat = convolve_fft(mask_sat, kernel)
        mask_sat = mask_sat >= 0.99 # boolean

        # creating a list of tuples which are the (row,column) coords of each science pixel
        rows = list(range(self.rampy_cube.shape[1]))
        cols = list(range(self.rampy_cube.shape[2]))

        pixels = []

        for i in rows:
            row_num = [i] * len(cols)
            pixel_row = list(zip(row_num,cols)) # tuples of a single row's (i's) pixel coordinates
            pixels.extend(pixel_row)

        if self.subarray == 'FULL':
            self.mask_tot = mask_sat & mask
            if self.no_sat_mask:
                self.mask_tot = mask
        else:
            self.mask_tot = mask_sat # need to add option here

        nan_mask = self.mask_tot * 1.0 
        nan_mask[nan_mask < 1] = np.nan
        self.nan_mask = nan_mask

        pixel_mask = self.mask_tot.flatten(order='C').tolist() # flattening mask to make same size/dimensions as the list of pixel coords
        self.masked_pixels = [pixel for pixel, m in zip(pixels, pixel_mask) if m]


    def _pixel_integration(self,cube,int_num,row,col):
        """
        Gets the x and y data of a specified integration for a specific pixel in a specified cube
        """
        integration_length = list(range(0,self.n_group))
        x_dat = [i + (int_num)*self.n_group for i in integration_length]
    
        ramp = []
        for x in x_dat:
            ramp.append(cube[x][row][col])

        return x_dat, ramp   


    def parallel_fit_df(self,cube,save_df=False):
        """
        fits all ramps of specified cube parallely
        """
        fitting = Parallel(n_jobs=self.num_cores, verbose=0)(
            delayed(linear_fitting)(pixel,cube,self.n_int,self.n_group) for pixel in self.masked_pixels)

        obj_df = pd.DataFrame(fitting, columns=["row","col","gradients","intercepts","residuals"])
        obj_df['max_residual'] = obj_df['residuals'].apply(max)
        obj_df['mean_residual'] = obj_df['residuals'].apply(np.mean)

        self.obj_df = obj_df
        if save_df:
            filepath = os.path.join(self.obs_dir, 'ramp_fittings.csv')
            self.obj_df.to_csv(filepath, index=False)


    def _line(self,m,c,x):
        """
        straight line eqn
        """
        return [i*m + c for i in x]


    def _check_jump(self,coords):
        """
        checking if jump was detected in the dq cube
        """
        row, col = coords
        # check through all frames for jumps  
        vals = self.jump_cube[:, row, col]
        if vals.any():
            return 1, int(np.argmax(vals))  # 1 and first z index
        else:
            return 0, None


# --------------------- Image Search -----------------------


    def mega_inator(self,cube):
        """
        makes a mega cube out of a rampy one
        """
        frames = list(range(0,self.n_frame))
        integrations = list(range(0,self.n_int))
        int_frames = [i*self.n_group  for i in integrations]

        int_frames.pop(0) # in order to have list of indices of first frames in ramps except for the very first one
        end_frames = [((i+1)*self.n_group)-1 for i in integrations] # want to remove the frames with these indices

        mega_cube = np.zeros((self.n_frame,cube.shape[1],cube.shape[2]))
        difference = 0 # difference between end of one ramp and start of next
        zero_ff = 0 # fudge factor to zero the ramps

        for frame in frames:
            if frame == 0:
                zero_ff = cube[frame]
                mega_cube[frame] = cube[frame] - zero_ff
            elif frame in int_frames:
                zero_ff = cube[frame] 
                difference = mega_cube[frame-2] + 2*(mega_cube[frame-2] - mega_cube[frame-3])
                mega_cube[frame] = cube[frame] + difference - zero_ff
            else:
                mega_cube[frame] = cube[frame] + difference - zero_ff

        # recalc int_frames
        int_frames = [i * self.n_group for i in integrations]
        end_frames = [(i+1) * self.n_group -1 for i in integrations]
        int_frames.extend(end_frames)

        mega_cube_masked = mega_cube.copy()
        nans_frame = np.full_like(mega_cube_masked[0], np.nan)

        for frame in int_frames:
            mega_cube_masked[frame] = nans_frame

        self.mega_cube = mega_cube
        self.mega_cube_masked = mega_cube_masked


    def _cube_gradient(self,cube,save=None):
        """
        make a gradient cube with the fakey fake frames for mega method
        for ramp method just takes the gradient then masks out bad frames
        """
        if self.method == 'mega':
            fakeified_cube = cube.copy()
            for int_num in range(self.n_int-1): # for all integrations but the last one
                val = (int_num+1)*self.n_group
                fakeified_cube[val-1] = 2*fakeified_cube[val-2] - fakeified_cube[val-3] # last frame of the integration
                fakeified_cube[val] = 3*fakeified_cube[val-2] - 2*fakeified_cube[val-3] # first frame of the next integration
            self.fakey_cube = fakeified_cube
            self.grad_cube = np.gradient(fakeified_cube,axis=0)

        if self.method == 'ramp':
            grad_cube = np.gradient(cube,axis=0)
            grad_cube[self.bad_frames] = np.nan
            self.grad_cube = grad_cube

        if save:
            filepath = os.path.join(self.obs_dir, "grad_cube.npy")
            np.save(filepath, self.grad_cube)


    def _reference_frame(self):
        """
        Making a reference frame but more complicated to counteract smearing
        of bright asteroids. - Armin's suggestion (starts with original, then masks)
        """
        # Use NaN fraction to make sure not a mostly NaN frame
        nan_fraction = np.isnan(self.grad_cube).reshape(self.n_frame, -1).mean(axis=1)
        not_nans = nan_fraction < 0.1
        not_nans[self.bad_frames] = False

        good_slices = self.grad_cube.copy()[not_nans]

        if len(good_slices) == 0:
            raise RuntimeError("No valid frames found for reference frame — "
                            "check grad_cube for all-NaN output.")

        self.first_ref_frame = np.nanmedian(good_slices, axis=0)

        filepath = os.path.join(self.obs_dir, "ref_frame_1.npy")
        np.save(filepath, self.first_ref_frame)


    def _cube_differenced(self,cube,reference,save=None,first=None):
        """
        make a differenced cube from gradient cube using a median frame as reference
        """
        diff_cube = cube.copy() - reference[np.newaxis,:,:]
        diff_cube[self.bad_frames] = np.nan

        self.diff_cube = diff_cube
        self.diff_cube_masked = self.diff_cube.copy() * self.mask_tot

        if save:
            filepath = os.path.join(self.obs_dir, "diff_cube.npy")
            np.save(filepath, self.diff_cube)

        if first:
            filepath = os.path.join(self.obs_dir, "diff_cube_1.npy")
            np.save(filepath, self.diff_cube)


    def _psf_kernel(self,size=10):
        """
        creates kernel based on filter using stpsf
        """
        miri = stpsf.MIRI()
        miri.filter = self.filter
        psf = miri.calc_psf(fov_pixels=size)
        self.kernel = psf[3].data

    
    def source_extracting(self,cube,save_plot,save_csv):
        """
        using source extractor (sep) instead of StarFinder
        """
        tasks = (delayed(_run_sep)(frame,cube[frame],self.kernel,self.mask_tot,save_plot,self.obs_dir,self.n_group)
                                                    for frame in range(self.n_frame))
        
        # run sep in parallel
        results = Parallel(n_jobs=self.num_cores, prefer="processes")(tasks)
        obj_dfs, filtered_dfs = zip(*results)

        # keep only non empty dfs
        non_empty_obj = [df for df in obj_dfs if not df.empty]
        non_empty_filt = [df for df in filtered_dfs if not df.empty]

        # total sep detections
        if len(non_empty_obj) == 0:
            self.total_df = pd.DataFrame(columns=obj_dfs[0].columns)
        else:
            self.total_df = pd.concat(non_empty_obj, ignore_index=True)

        # filtered SEP detections 
        if len(non_empty_filt) == 0:
            self.filtered_sep_df = pd.DataFrame(columns=filtered_dfs[0].columns)
        else:
            self.filtered_sep_df = pd.concat(non_empty_filt, ignore_index=True)

        # printing detection stats
        print(f"SEP: {len(non_empty_filt)} / {self.n_frame} frames with filtered detections "
              f"({len(self.filtered_sep_df)} total sources)")

        # save csv's of the filtered and unfiltered dfs
        if save_csv:
            if len(self.filtered_sep_df) > 0:
                filepath = os.path.join(self.obs_dir, "filtered_sources.csv")
                self.filtered_sep_df.to_csv(filepath, index=False)

            filepath = os.path.join(self.obs_dir, "all_sources.csv")
            self.total_df.to_csv(filepath, index=False)


    def _masked_reference(self,mask_correction,mask_radius=10):
        """
        Makes a reference frame (median) but masks out any variable sources.
        Masks detected source positions and takes nanmedian of remaining pixels.
        """
        if self.filtered_sep_df.empty:
            self.second_ref_frame = self.first_ref_frame.copy()
            return

        if mask_correction == False:
            self.second_ref_frame = self.first_ref_frame.copy()
            return

        # source masks for each frame with detected variables
        reference_cube = np.zeros_like(self.grad_cube)
        kernel = self._circle_app(mask_radius)

        for frame in self.frames:
            if frame in self.filtered_sep_df['frame'].values:
                mask = np.zeros_like(self.grad_cube[0])

                frame_df = self.filtered_sep_df[self.filtered_sep_df['frame'] == frame]
                x_int = [round(x) for x in frame_df['x'].values]
                y_int = [round(y) for y in frame_df['y'].values]

                for i in range(len(x_int)):
                    mask[y_int[i], x_int[i]] = 1

                reference_cube[frame] = convolve_fft(mask, kernel)

        source_mask = reference_cube >= 0.00001  # boolean: True = source pixel to exclude
        self.source_mask = source_mask

        # Only use good frames (not bad/all-NaN)
        nan_fraction = np.isnan(self.grad_cube).reshape(self.n_frame, -1).mean(axis=1)
        not_nans = nan_fraction < 0.1
        not_nans[self.bad_frames] = False

        good_slices = self.grad_cube.copy()[not_nans]
        mask_slices = self.source_mask[not_nans]

        # NaN out source pixels, then make reference from median
        masked_slices = np.where(mask_slices, np.nan, good_slices)
        self.second_ref_frame = np.nanmedian(masked_slices, axis=0)

        filepath = os.path.join(self.obs_dir, "ref_frame_2.npy")
        np.save(filepath, self.second_ref_frame)


    def _remove_cosmic(self,cube):
        """
        uses lacosmic to remove the cosmic rays in each frame
        """
        # run lacosmic on each frame in parallel
        results = Parallel(n_jobs=self.num_cores,verbose=0)(delayed(run_lacosmic)(cube[i],self.mask_tot) for i in range(len(cube)))
        clean_cube = np.array([r[0] for r in results])
        cr_mask_cube = np.array([r[1] for r in results])

        self.cr_mask_cube = cr_mask_cube
        self.clean_cube = clean_cube

        filepath = os.path.join(self.obs_dir, "clean_cube.npy")
        np.save(filepath, self.clean_cube)


    def _make_ref_cr_mask(self):
        """
        makes a cosmic ray mask that is a union of the lacosmic cr_mask
        and the JWST pipeline jump detections from the dq array
        """
        ref_cr_mask = self.cr_mask_cube | self.jump_cube
        self.ref_cr_mask = ref_cr_mask


# --------------------- Significance Functions -----------------------


    def _cube_significance(self,magic_number=3):
        """
        making a significance cube - dividing the differenced cube by the 
        standard deviation of the background of each frame
        Then making a cut based on a ~magic number~ which at this point is just 3
        """
        frame_num = list(range(0, self.n_frame))
        sig_cube = np.zeros_like(self.diff_cube)

        dat = np.where(self.mask_tot[None, :, :], self.clean_cube, np.nan)

        for frame in frame_num:
            data = dat[frame].copy()
            _, med, std = sigma_clipped_stats(data.copy())

            sig_cube[frame] = (dat[frame]-med) / std

        # compare with the cr ref mask and set to zero where mask is
        sig_cube[self.ref_cr_mask] = 0
        
        self.sig_cube = sig_cube
        self.bool_sig_cube = sig_cube > magic_number


    def _cube_threshold(self,rad=2,threshold=9):
        """
        convolves the significance cube with a circle and identifies
        the bits above a threshold, above which should be psf-like sources
        and below are cosmic ray junk stuffs (ideally)
        """
        frame_num = list(range(0, self.n_frame))
        conv_sig_cube = np.zeros_like(self.sig_cube)
        bool_threshold_cube = np.zeros_like(self.sig_cube)

        for frame in frame_num:
            conv_sig_cube[frame] = convolve_fft(self.bool_sig_cube[frame],self._circle_app(rad),normalize_kernel=False)
            bool_threshold_cube[frame] = conv_sig_cube[frame] > threshold

        self.conv_sig_cube = conv_sig_cube
        self.bool_threshold_cube = bool_threshold_cube


    def _cube_rolling_sum(self,num_frames=4,threshold=3):
        """
        rolling sum over (num_frames) frames of threshold cube, cut for >= threshold
        to identify 'significant' flux changes
        """
        good_frames_cube = np.delete(self.bool_threshold_cube, self.bad_frames, axis=0)
        n_good = good_frames_cube.shape[0]
        rows = good_frames_cube.shape[1]
        cols = good_frames_cube.shape[2]

        rolling_sum_cube = np.zeros((n_good,rows,cols), dtype=int)

        for frame in range(n_good):
            rolling_sum_cube[frame] = np.sum(good_frames_cube[frame:frame+num_frames], axis=0)

        # make cut for >= threshold
        bool_rolling_sum_cube = rolling_sum_cube >= threshold

        # reinsert bad frames at their original positions by pre-allocating
        # the full arrays and index-assigning the good frame values
        n_total = self.bool_threshold_cube.shape[0]
        good_idx = [i for i in range(n_total) if i not in set(self.bad_frames)]

        rolling_sum_full = np.full((n_total, rows, cols), np.nan)
        bool_rolling_sum_full = np.zeros((n_total, rows, cols), dtype=bool)
        rolling_sum_full[good_idx] = rolling_sum_cube
        bool_rolling_sum_full[good_idx] = bool_rolling_sum_cube

        rolling_sum_cube = rolling_sum_full
        bool_rolling_sum_cube = bool_rolling_sum_full

        self.rolling_sum_cube = rolling_sum_cube
        self.bool_rolling_sum_cube = bool_rolling_sum_cube

        filepath = os.path.join(self.obs_dir, "rolling_sum_cube.npy")
        np.save(filepath, self.rolling_sum_cube)

    
    def _significance_output(self):
        """
        Making the output for the significance way of things
        For now makes (and saves?) a dataframe containing the pixel coords and frame
        where something has passed the multiple signicance thresholds.
        """
        frames,rows,cols = np.where(self.bool_rolling_sum_cube==True)
        data_dict = {'frame': frames,
                     'x': cols,
                     'y': rows}
        
        significance_df = pd.DataFrame(data_dict)
        self.significance_df = significance_df
        
        if len(self.significance_df) > 0:
            filepath = os.path.join(self.obs_dir, 'significance.csv')
            significance_df.to_csv(filepath,index=False)


# ------------------------------ Output stuff! ----------------------------
    

    def _spatial_group(self, df, min_samples=1, distance=1):
        """
        Groups events based on proximity w/ dbscan
        """
        if df.empty:
            df['objid'] = pd.Series(dtype=int)
            return df
        
        output = df.copy()

        pos = np.column_stack([output['x'].values, output['y'].values])
        cluster = DBSCAN(eps=distance, min_samples=min_samples, n_jobs=self.num_cores).fit(pos)
        labels = cluster.labels_
        objid = labels + 1
        objid[objid < 0] = 0 
        output['objid'] = objid.astype(int)

        return output
    

    def _temporal_group(self,df,min_samples=5,distance=2):
        """
        Groups events based on time w/ dbscan
        """
        if df.empty:
            return df

        output = pd.DataFrame()

        ids_list = sorted(df['objid'].unique())

        for id in ids_list:
            obj_df = df[df['objid']==id]
            # loop through each grouped object to find events
            if len(obj_df) >= 3:
                data = obj_df['frame'].values
                data = data.reshape(-1, 1)
                db = DBSCAN(eps=distance, min_samples=min_samples,n_jobs=self.num_cores).fit(data)
                labels = db.labels_
                obj_df['event'] = labels.astype(int)

            output = pd.concat([output, obj_df], ignore_index=True)

        # see if can clean up number of events    
        if len(output) > 0:
            filepath = os.path.join(self.obs_dir, f'ms-{min_samples}_d-{distance}_events.csv')
            output.to_csv(filepath,index=False)

        return output


    def asteroid_candidate(self, df, threshold_1=10, threshold_2=5, threshold_3=2):
        """
        Determines if in the grouped detections there are any potential asteroids.
        Uses trajectory-based classification from _tag_asteroids when available,
        otherwise falls back to displacement grading.
        """
        ids = []

        if 'asteroid_id' in df.columns and (df['asteroid_id'] > 0).any():
            for ast_id in sorted(df[df['asteroid_id'] > 0]['asteroid_id'].unique()):
                obj = df[df['asteroid_id'] == ast_id].sort_values('frame')
                objid = int(obj['objid'].iloc[0])
                x0, y0 = obj.iloc[0]['x'], obj.iloc[0]['y']
                ids.append(f'Asteroid ID {ast_id}, Object: {objid}, '
                           f'Start Coords: ({x0:.2f},{y0:.2f})')
            return len(ids), ids

        # Fallback: displacement grading
        num_candidates_1 = 0
        num_candidates_2 = 0
        num_candidates_3 = 0

        for id in range(1, df['objid'].max() + 1):
            df_obj = df[df['objid'] == id]
            if len(df_obj) < 2:
                continue
            try:
                idx_min = df_obj['frame'].idxmin()
                idx_max = df_obj['frame'].idxmax()
            except ValueError:
                continue
            row_min = df_obj.loc[idx_min]
            row_max = df_obj.loc[idx_max]
            dist = np.sqrt((row_max['x'] - row_min['x'])**2 +
                           (row_max['y'] - row_min['y'])**2)
            if dist > threshold_1:
                num_candidates_1 += 1
                ids.append(f'Grade 1, Object: {id}, Start Coords: ({row_min["x"]:.2f},{row_min["y"]:.2f})')
            elif dist > threshold_2:
                num_candidates_2 += 1
                ids.append(f'Grade 2, Object: {id}, Start Coords: ({row_min["x"]:.2f},{row_min["y"]:.2f})')
            elif dist > threshold_3:
                num_candidates_3 += 1
                ids.append(f'Grade 3, Object: {id}, Start Coords: ({row_min["x"]:.2f},{row_min["y"]:.2f})')

        return num_candidates_1 + num_candidates_2 + num_candidates_3, ids


    def _tag_asteroids(self, df, min_frames=5, min_displacement_px=2.0,
                       max_residual_px=3.0, link_eps_px=5.0, min_track_frames=10):
        """
        Classifies objects as asteroids by fitting a linear trajectory (x, y vs mjd).
        Objects with significant displacement and low trajectory residuals are flagged.
        Nearby objects that fall on the same trajectory are linked with a shared asteroid_id.
        Adds 'classification' and 'asteroid_id' columns.
        """
        from scipy import stats as scipy_stats

        df = df.copy()
        df['classification'] = 'Unknown'
        df['asteroid_id'] = -1

        objids = sorted([oid for oid in df['objid'].unique() if oid > 0])
        tracks = {}

        for objid in objids:
            obj = df[df['objid'] == objid].sort_values('mjd')
            if len(obj) < min_frames:
                continue
            x_vals = obj['x'].values
            y_vals = obj['y'].values
            mjd_vals = obj['mjd'].values
            dist = np.sqrt((x_vals[-1] - x_vals[0])**2 + (y_vals[-1] - y_vals[0])**2)
            if dist < min_displacement_px:
                continue
            t_ref = mjd_vals.mean()
            t = mjd_vals - t_ref
            slope_x, intercept_x, *_ = scipy_stats.linregress(t, x_vals)
            slope_y, intercept_y, *_ = scipy_stats.linregress(t, y_vals)
            pred_x = intercept_x + slope_x * t
            pred_y = intercept_y + slope_y * t
            rms = np.sqrt(np.mean((x_vals - pred_x)**2 + (y_vals - pred_y)**2))
            if rms < max_residual_px:
                tracks[objid] = dict(slope_x=slope_x, intercept_x=intercept_x,
                                     slope_y=slope_y, intercept_y=intercept_y,
                                     t_ref=t_ref, rms=rms)

        if not tracks:
            return df

        # Link nearby non-asteroid objids onto existing trajectories
        asteroid_objids = set(tracks.keys())
        other_objids = set(objids) - asteroid_objids
        assigned = {oid: oid for oid in asteroid_objids}

        for other_oid in other_objids:
            obj = df[df['objid'] == other_oid]
            x_c = obj['x'].mean()
            y_c = obj['y'].mean()
            t_c = obj['mjd'].mean()
            for leader_oid, tr in tracks.items():
                dt = t_c - tr['t_ref']
                pred_x = tr['intercept_x'] + tr['slope_x'] * dt
                pred_y = tr['intercept_y'] + tr['slope_y'] * dt
                if np.sqrt((x_c - pred_x)**2 + (y_c - pred_y)**2) < link_eps_px:
                    assigned[other_oid] = leader_oid
                    break

        leaders = sorted(set(assigned.values()))
        id_map = {leader: i + 1 for i, leader in enumerate(leaders)}

        for oid, leader in assigned.items():
            mask = df['objid'] == oid
            if mask.sum() >= min_track_frames:
                df.loc[mask, 'classification'] = 'Asteroid'
                df.loc[mask, 'asteroid_id'] = id_map[leader]

        return df


    def _time_mjd(self):
        """
        takes df with col 'frame' and adds a mjd col
        the time of the frames in mjd
        """
        df = self.time_df
        df = df.apply(lambda s: s.astype(s.dtype.newbyteorder('=')))

        frames = self.frames
        times = []

        for i in range(len(df)):
            start = df.loc[i, "int_start_MJD_UTC"]
            end   = df.loc[i, "int_end_MJD_UTC"]
            times.extend(np.linspace(start,end,self.n_group))

        data = {'frame': frames, 'mjd': times}

        self.frame_mjd_df = pd.DataFrame(data) 


    def assign_mjd(self,df):
        """
        for a given pd dataframe with a column 'frame' will assign a mjd column
        """
        df = df.merge(self.frame_mjd_df, on="frame", how="left")

        return df


    def plot_detection(self, save_dir, latex=True, lc_units='dn/s'):
        """
        Tessellate-identical 3-panel figure for each detected event:
          left   — light curve with event span highlighted and zoom inset
          middle — 19x19 px cutout at the brightest frame ('Brightest image')
          right  — same cutout 1 hour later
        Saves as object{objid:04d}_event{eventid}of{total_events}.png
        """
        import matplotlib.patches as patches
        from matplotlib.lines import Line2D
        from mpl_toolkits.axes_grid1.inset_locator import mark_inset

        # if latex:
        #     plt.rc('text', usetex=True)

        os.makedirs(save_dir, exist_ok=True)

        mjd_arr = self.frame_mjd_df.set_index('frame')['mjd'].values
        time = mjd_arr - mjd_arr[0]
        cadence = np.median(np.diff(time))
        group_time_s = cadence * 86400  # MJD days → seconds, for DN/s scaling

        # Detect gaps in the time series
        med = np.nanmedian(np.diff(time))
        std = np.nanstd(np.diff(time))
        break_ind = np.where(np.diff(time) > med + 1 * std)[0]
        break_ind = np.append(break_ind, len(time))
        break_ind += 1
        break_ind = np.insert(break_ind, 0, 0)

        obj_ids = sorted([oid for oid in self.events['objid'].unique() if oid > 0])

        for objid in obj_ids:
            obj_df = self.events[self.events['objid'] == objid].sort_values('frame')
            if obj_df.empty:
                continue

            # Build per-object event list from DBSCAN 'event' column
            if 'event' in obj_df.columns and obj_df['event'].notna().any():
                event_labels = sorted(obj_df['event'].dropna().unique())
            else:
                event_labels = [None]
            total_events = len(event_labels)

            for eventid, event_label in enumerate(event_labels, start=1):
                ev_df = obj_df if event_label is None else obj_df[obj_df['event'] == event_label]
                if ev_df.empty:
                    continue

                x = int(round(ev_df['x'].mean()))
                y = int(round(ev_df['y'].mean()))
                frame_start = int(ev_df['frame'].min())
                frame_end = int(ev_df['frame'].max())

                # Aperture LC using filter FWHM as radius (buffer = floor(fwhm))
                buf = int(np.floor(self.fwhm))

                aperture = self.clean_cube[:,
                                           max(0, y - buf):min(self.clean_cube.shape[1], y + buf + 1),
                                           max(0, x - buf):min(self.clean_cube.shape[2], x + buf + 1)]
                all_nan = np.all(np.isnan(aperture), axis=(1, 2))
                f = np.where(all_nan, np.nan, np.nansum(aperture, axis=(1, 2)))
                if lc_units == 'dn/s':
                    f = f / group_time_s

                # Brightest frame within detection span
                if frame_end - frame_start >= 2:
                    brightestframe = frame_start + int(np.where(
                        np.abs(f[frame_start:frame_end]) == np.nanmax(np.abs(f[frame_start:frame_end]))
                    )[0][0])
                else:
                    brightestframe = frame_start
                try:
                    brightestframe = int(brightestframe)
                except TypeError:
                    brightestframe = int(brightestframe[0])
                if brightestframe >= self.clean_cube.shape[0]:
                    brightestframe -= 1
                if frame_end >= self.clean_cube.shape[0]:
                    frame_end -= 1

                fstart = frame_start - 20
                if fstart < 0:
                    fstart = 0
                zoom = f[fstart:frame_end + 20]

                fig, ax = plt.subplot_mosaic([[1, 1, 1, 2, 2], [1, 1, 1, 3, 3]],
                                             figsize=(7 * 1.1, 5.5 * 1.1), constrained_layout=True)

                # Ghost plot to fix inset ylims
                ax[1].plot(time[fstart:frame_end + 20], zoom, 'k', alpha=0)
                insert_ylims = ax[1].get_ylim()

                # Full light curve per segment
                for seg in range(len(break_ind) - 1):
                    ax[1].plot(time[break_ind[seg]:break_ind[seg + 1]],
                               f[break_ind[seg]:break_ind[seg + 1]], 'k', alpha=0.8)

                ylims = ax[1].get_ylim()
                ax[1].set_ylim(ylims[0], ylims[1] + abs(ylims[0] - ylims[1]))
                ax[1].set_xlim(np.min(time), np.max(time))
                ax[1].set_title(f'ObjID: {objid}', fontsize=15)
                ax[1].set_ylabel('DN/s' if lc_units == 'dn/s' else 'DN/group', fontsize=15, labelpad=10)
                ax[1].set_xlabel(f'Time (MJD - {np.round(mjd_arr[0], 3)})', fontsize=15)

                axins = ax[1].inset_axes([0.1, 0.55, 0.86, 0.43])
                axins.axvspan(time[frame_start] - cadence / 2,
                              time[frame_end] + cadence / 2, color='C1', alpha=0.4)
                for seg in range(len(break_ind) - 1):
                    axins.plot(time[break_ind[seg]:break_ind[seg + 1]],
                               f[break_ind[seg]:break_ind[seg + 1]], 'k', alpha=0.8, marker='.')

                duration = frame_end - frame_start
                if duration < 4:
                    duration = 4
                fe = frame_end + 20
                if fe >= len(time):
                    fe = len(time) - 1
                xmin_z = time[frame_start] - (3 * duration * cadence)
                xmax_z = time[frame_end] + (3 * duration * cadence)
                if xmin_z <= 0:
                    xmin_z = 0
                if xmax_z >= np.nanmax(time):
                    xmax_z = np.nanmax(time)
                axins.set_xlim(xmin_z, xmax_z)
                axins.set_ylim(insert_ylims[0], insert_ylims[1])
                mark_inset(ax[1], axins, loc1=3, loc2=4, fc="none", ec="r", lw=2)
                plt.setp(axins.spines.values(), color='r', lw=2)
                plt.setp([axins.get_xticklines(), axins.get_yticklines()], color='C3')

                # Colour stretch from 3x3 patch at brightest frame
                bright_frame = self.clean_cube[brightestframe, max(0, y - 1):y + 2, max(0, x - 1):x + 2]
                vmin = np.percentile(self.clean_cube[brightestframe], 16)
                try:
                    vmax = np.percentile(bright_frame, 80)
                except Exception:
                    vmax = vmin + 20
                if vmin >= vmax:
                    vmin = vmax - 5

                # 19x19 px cutout
                ymin = y - 9
                if ymin < 0:
                    ymin = 0
                xmin = x - 9
                if xmin < 0:
                    xmin = 0
                cutout_image = self.clean_cube[:, ymin:y + 10, xmin:x + 10]

                ax[2].imshow(cutout_image[brightestframe], cmap='gray', origin='lower',
                             vmin=vmin, vmax=vmax)
                ax[2].scatter(ev_df['x'].mean() - xmin, ev_df['y'].mean() - ymin,
                              color='r', s=50, marker='x', lw=2)
                ax[2].set_title('Brightest frame', fontsize=15)
                ax[2].get_xaxis().set_visible(False)
                ax[2].get_yaxis().set_visible(False)
                ax[3].get_xaxis().set_visible(False)
                ax[3].get_yaxis().set_visible(False)

                # 20 non-NaN frames after; fall back to 20 non-NaN frames before
                valid_after = [i for i in range(brightestframe + 1, len(cutout_image))
                               if not np.all(np.isnan(cutout_image[i]))]
                if len(valid_after) >= 20:
                    after = valid_after[19]
                else:
                    valid_before = [i for i in range(brightestframe)
                                    if not np.all(np.isnan(cutout_image[i]))]
                    after = valid_before[-20] if len(valid_before) >= 20 else (valid_before[0] if valid_before else brightestframe)

                offset = after - brightestframe
                after_label = f'+{offset}' if offset >= 0 else str(offset)

                ax[3].imshow(cutout_image[after], cmap='gray', origin='lower',
                             vmin=vmin, vmax=vmax)
                ax[3].set_title(f'Frame {after_label}', fontsize=15)
                ax[3].annotate('', xy=(0.2, 1.15), xycoords='axes fraction', xytext=(0.2, 1.),
                               arrowprops=dict(arrowstyle="<|-", color='r', lw=3))
                ax[3].annotate('', xy=(0.8, 1.15), xycoords='axes fraction', xytext=(0.8, 1.),
                               arrowprops=dict(arrowstyle="<|-", color='r', lw=3))

                # 5x5 red/cyan detection box — separate patches per panel (matching tessellate)
                rect = patches.Rectangle((x - 2.5 - xmin, y - 2.5 - ymin), 5, 5,
                                         linewidth=3, edgecolor='r', facecolor='none')
                ax[2].add_patch(rect)
                ax[2].add_line(Line2D([x - 2.5 - xmin, x + 2.5 - xmin],
                                      [y + 2.5 - ymin, y + 2.5 - ymin], color='c', linewidth=3))
                ax[2].add_line(Line2D([x + 2.5 - xmin, x + 2.5 - xmin],
                                      [y - 2.5 - ymin, y + 2.5 - ymin], color='c', linewidth=3))

                rect = patches.Rectangle((x - 2.5 - xmin, y - 2.5 - ymin), 5, 5,
                                         linewidth=3, edgecolor='r', facecolor='none')
                ax[3].add_patch(rect)
                ax[3].add_line(Line2D([x - 2.5 - xmin, x + 2.5 - xmin],
                                      [y + 2.5 - ymin, y + 2.5 - ymin], color='c', linewidth=3))
                ax[3].add_line(Line2D([x + 2.5 - xmin, x + 2.5 - xmin],
                                      [y - 2.5 - ymin, y + 2.5 - ymin], color='c', linewidth=3))

                plt.savefig(os.path.join(save_dir,
                                         f'object{objid:04d}_event{eventid}of{total_events}.png'),
                            bbox_inches='tight')
                plt.close(fig)


    def save_outputs(self):
        """
        outputs i wanna save (from themselves)
        """
        def safe_max(series):
            """
            Returns max if there is one, returns 0 otherwise
            """
            if series is None:
                return 0
            if not hasattr(series, "__len__"):
                return 0
            if len(series) == 0:
                return 0
            arr = series.to_numpy()
            if arr.size == 0:
                return 0
            return np.nanmax(arr)

        # make a folder for the grouped output
        self.grouped_dir = os.path.join(self.obs_dir, 'grouped_output')
        os.makedirs(self.grouped_dir, exist_ok=True)
        
        if len(self.significance_df) > 0:
            g_sig_df = self._spatial_group(self.significance_df)
            g_sig_df = g_sig_df.sort_values(by=['objid', 'frame'], ascending=[True, True])
            g_sig_df = self._temporal_group(g_sig_df)
            g_sig_df = self.assign_mjd(g_sig_df)
            filepath = os.path.join(self.grouped_dir, 'grouped_significance.csv')
            g_sig_df.to_csv(filepath, index=False)
            if self.plot and len(g_sig_df) > 0:
                if not hasattr(self, 'events'):
                    self.events = g_sig_df
                self.plot_detection(os.path.join(self.grouped_dir, 'detection_figures_sig'))

        # filtered sep sources (grouped)
        num_candidates, data = 0, []
        if len(self.filtered_sep_df) > 0:
            self.events = self._spatial_group(self.filtered_sep_df)
            self.events = self.events.sort_values(by=['objid', 'frame'], ascending=[True, True])
            self.events = self._temporal_group(self.events)
            self.events = self.assign_mjd(self.events)
            self.events = self._tag_asteroids(self.events)
            self.events.to_csv(os.path.join(self.grouped_dir, 'grouped_filtered_sep.csv'), index=False)
            if self.plot and len(self.events) > 0:
                self.plot_detection(os.path.join(self.grouped_dir, 'detection_figures_sep'))
            num_candidates, data = self.asteroid_candidate(self.events)

        if len(self.total_df) > 0:
            g_tot_sep_df = self._spatial_group(self.total_df)
            g_tot_sep_df = g_tot_sep_df.sort_values(by=['objid', 'frame'], ascending=[True, True])
            g_tot_sep_df = self._temporal_group(g_tot_sep_df)
            g_tot_sep_df = self.assign_mjd(g_tot_sep_df)
        else:
            g_tot_sep_df = self.total_df.copy()
            g_tot_sep_df['objid'] = pd.Series(dtype=int)
        filepath = os.path.join(self.grouped_dir, 'grouped_total_sep.csv')
        g_tot_sep_df.to_csv(filepath, index=False)

        full_file_path = os.path.join(self.grouped_dir, 'objects_summary.txt')
        summary_path = os.path.join(self.base_dir, 'interesting_findings.txt')

        with open(summary_path, "a") as summary:
            if len(self.filtered_sep_df) > 0 and len(data) > 0:
                print(f"------------------------------------------------------", file=summary)
                print(f"{self.filename}", file=summary)
                print(f"------------------------------------------------------", file=summary)
                print(f"{num_candidates} asteroid candidates in filtered objects", file=summary)
                if len(data) >= 1:
                    for i in range(len(data)):
                        print(f"----- {data[i]}", file=summary)
                print(" ", file=summary)

        with open(full_file_path, "w") as f:
            print(f"{safe_max(g_tot_sep_df['objid'])} total objects identified by sep", file=f)
            if len(self.filtered_sep_df) > 0:
                print(f"{safe_max(self.events['objid'])} filtered objects identified by sep", file=f)
                print(f"----- {num_candidates} asteroid candidates in filtered objects", file=f)
                if len(data) >= 1:
                    for i in range(len(data)):
                        print(f"---------- {data[i]}", file=f)
            else:
                print('0 objects passed through filtering', file=f)
            if len(self.significance_df) > 0:
                print(f"{safe_max(g_sig_df['objid'])} objects identified by significance", file=f)
            else:
                print('0 objects identified by significance', file=f)

        print(f'{safe_max(g_tot_sep_df["objid"])} total objects identified by sep')
        if len(self.filtered_sep_df) > 0:
            print(f'{safe_max(self.events["objid"])} filtered objects identified by sep')
            print(f"----- {num_candidates} asteroid candidates in filtered objects")
            if len(data) >= 1:
                for i in range(len(data)):
                    print(f"---------- {data[i]}")
        else:
            print('0 objects passed through filtering')
        if len(self.significance_df) > 0:
            print(f'{safe_max(g_sig_df["objid"])} objects identified by significance')
        else:
            print('0 objects identified by significance')


for file in glob.glob('/home/phys/astronomy/jlu69/Masters/jurassic/pipeline_data/Obs/stage1/ast6/*ramp.fits'):
    Jurassic(file, method='mega', num_cores=55)