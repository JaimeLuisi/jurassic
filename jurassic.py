import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
try:
    import stpsf
except ModuleNotFoundError:
    import webbpsf as stpsf
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

def _moment_elongation(img):
    """
    SEP-style elongation e = sqrt((Ixx-Iyy)^2 + 4*Ixy^2) / (Ixx+Iyy) of an
    image, from non-negative-weighted second moments about its own
    centroid. 0 = circularly symmetric, -> 1 = a line. Used to compare a
    candidate's actual shape against what the model PSF predicts at the
    same fit offset (see _psf_fit's elongation_excess).
    """
    ny, nx = img.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    w = np.clip(img, 0, None)
    wsum = w.sum()
    if wsum <= 0:
        return np.nan
    xc = (w * xx).sum() / wsum
    yc = (w * yy).sum() / wsum
    Ixx = (w * (xx - xc) ** 2).sum() / wsum
    Iyy = (w * (yy - yc) ** 2).sum() / wsum
    Ixy = (w * (xx - xc) * (yy - yc)).sum() / wsum
    denom = Ixx + Iyy
    if denom <= 0:
        return np.nan
    return float(np.sqrt((Ixx - Iyy) ** 2 + 4 * Ixy ** 2) / denom)


def _psf_fit(data_sub, x, y, kernel):
    """
    Fit PSF centroid by minimizing sum((norm_stamp - shifted_psf)**2) via Powell.
    Returns (psf_like, x_psf, y_psf, elongation_excess, symmetry180):
      psf_like           — Pearson r between stamp and best-fit shifted PSF.
                           A poor point-source discriminant on its own:
                           Pearson correlation is scale/shift invariant, so
                           an elongated cosmic-ray track can still correlate
                           well with a round PSF template as long as the
                           coarse "bright middle, faint edges" pattern lines
                           up — it doesn't penalize the actual shape mismatch.
      x_psf              — refined centroid x in image coordinates
      y_psf              — refined centroid y in image coordinates
      elongation_excess  — the core's own second-moment elongation (SEP-style
                           e, 0=round) minus the elongation the model PSF
                           shows at the same fit offset. ~0 for a real point
                           source, distinctly positive for an elongated
                           track — catches stretched-but-still-centered
                           features that fool both psf_like and symmetry180
                           (an elongated ellipse can still be inversion
                           symmetric).
      symmetry180        — Pearson r between the stamp core and its own
                           180-degree rotation about the fit center. Needs no
                           PSF model at all: a real (isotropic) PSF is
                           symmetric under 180-degree rotation by
                           construction, while a directional/comet-tail
                           feature generally isn't. Exact (no interpolation)
                           since the core is an odd-sized window centered on
                           an integer pixel. Complementary to
                           elongation_excess: catches lopsided/asymmetric
                           tracks that a symmetric-ellipse elongation measure
                           can miss.
    Returns (nan, x, y, nan, nan) if stamp is out of bounds or has no variance.
    """
    from scipy.optimize import minimize
    from scipy.ndimage import shift as ndshift

    h, w = data_sub.shape
    hs = kernel.shape[0] // 2
    xi, yi = int(round(x)), int(round(y))
    y0, y1 = yi - hs, yi + hs + 1
    x0, x1 = xi - hs, xi + hs + 1
    if y0 < 0 or x0 < 0 or y1 > h or x1 > w:
        return np.nan, x, y, np.nan, np.nan

    stamp = data_sub[y0:y1, x0:x1].astype(np.float64)
    stamp_sub = stamp - np.nanmedian(stamp)
    total = np.nansum(stamp_sub)
    if total == 0:
        return np.nan, x, y, np.nan, np.nan
    stamp_norm = stamp_sub / total

    psf_norm = kernel.astype(np.float64) / kernel.sum()

    # initial offset: fractional part of SEP centroid from rounded integer
    dx0 = x - xi
    dy0 = y - yi

    def residual(coeff):
        shifted = ndshift(psf_norm, (coeff[1], coeff[0]), order=3, mode='constant', cval=0)
        s = shifted.sum()
        if s > 0:
            shifted /= s
        return float(np.nansum((stamp_norm - shifted) ** 2)) * 1e6

    res = minimize(residual, [dx0, dy0], method='Powell',
                   bounds=[(-1.0, 1.0), (-1.0, 1.0)])
    dx_fit, dy_fit = res.x

    psf_shifted = ndshift(psf_norm, (dy_fit, dx_fit), order=3, mode='constant', cval=0)
    s = psf_shifted.sum()
    if s > 0:
        psf_shifted /= s

    # compute r on inner core only (~2×FWHM), not the full noisy stamp
    core_r = max(3, hs // 3)
    cy, cx = hs, hs
    core_sl = (slice(cy - core_r, cy + core_r + 1), slice(cx - core_r, cx + core_r + 1))
    stamp_f = stamp_norm[core_sl].flatten()
    psf_f = psf_shifted[core_sl].flatten()
    if stamp_f.std() == 0 or psf_f.std() == 0:
        return np.nan, x, y, np.nan, np.nan

    r = float(np.corrcoef(stamp_f, psf_f)[0, 1])

    # elongation excess: data's own shape vs. what the model PSF predicts
    # at this same fit offset, both measured identically (core window,
    # background-subtracted, non-negative-weighted moments)
    e_data = _moment_elongation(stamp_sub[core_sl])
    e_psf = _moment_elongation(total * psf_shifted[core_sl])
    elong_excess = e_data - e_psf if np.isfinite(e_data) and np.isfinite(e_psf) else np.nan

    # 180-degree self-symmetry: model-free, exact (odd-sized window, integer center)
    core = stamp_norm[core_sl]
    core_rot = core[::-1, ::-1]
    if core.std() == 0 or core_rot.std() == 0:
        sym180 = np.nan
    else:
        sym180 = float(np.corrcoef(core.flatten(), core_rot.flatten())[0, 1])

    return r, float(xi + dx_fit), float(yi + dy_fit), elong_excess, sym180


def _run_sep(frame, data, kernel, mask, save, obs_dir, n_group, psf_fwhm=None, psf_corr_thresh=0.8,
             edge_margin_x=16, edge_margin_y=12, cr_mask=None, cr_frame_window=1, cr_px_window=1):
    """
    run source extractor in parallel

    cr_mask : ndarray (n_frame, ny, nx) bool, optional
        lacosmic's cosmic-ray mask (Jurassic.cr_mask_cube — NOT the
        pipeline's JUMP_DET flag). Shape-based cuts (psf_like,
        elongation_excess, symmetry180) cannot reliably separate cosmic
        rays from real point sources on their own — a single-particle hit
        near-normal-incidence produces a compact, genuinely PSF-like charge
        cloud indistinguishable by shape from a real source at this
        detector's pixel scale. lacosmic's Laplacian test is a better
        per-detection reject signal because it asks a physically different
        question than shape correlation alone: is this pixel-scale feature
        consistent with real, telescope-optics-broadened light, or is it an
        unresolved detector-level spike? A real transient must satisfy the
        former regardless of its time behavior, so — unlike the pipeline's
        JUMP_DET, which flags *any* sudden ramp discontinuity and can't
        distinguish a cosmic ray from a genuine fast brightening — this
        doesn't risk rejecting real discoveries (verified by injection
        testing). It can still miss cosmic ray hits that happen to look
        PSF-like at this pixel scale, but a false negative here is a far
        safer failure mode than falsely rejecting a real transient. None
        disables the check (e.g. for the first-pass call, before the mask
        exists yet).
    cr_frame_window, cr_px_window : int
        Half-widths of the frame/pixel window checked around each
        candidate for a cr_mask hit.
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
        filtered_df = pd.DataFrame(columns=obj_df.columns.tolist() + ['symmetry', 'psf_like', 'frame', 'sep_flux', 'sep_fluxerr', 'sep_s/n', 'sep_flag'])
        return obj_df, filtered_df

    # adding needed cols
    obj_df['symmetry'] = (obj_df['a']/obj_df['b']).abs() - 1
    obj_df['frame'] = frame
    obj_df['group'] = (obj_df['frame'] % n_group) + 1 # adding the group number

    # PSF fit: centroid refinement + shape correlation
    psf_results = [_psf_fit(data_sub, row['x'], row['y'], kernel)
                   for _, row in obj_df.iterrows()]
    obj_df['psf_like']          = [r[0] for r in psf_results]
    obj_df['x_psf']             = [r[1] for r in psf_results]
    obj_df['y_psf']             = [r[2] for r in psf_results]
    obj_df['elongation_excess'] = [r[3] for r in psf_results]
    obj_df['symmetry180']       = [r[4] for r in psf_results]

    # aperture photometry
    flux, fluxerr, flag = sep.sum_circle(data_sub, obj_df['x'], obj_df['y'], 3.0, err=bkg.globalrms, gain=1.0) # ap radius = 3.0
    obj_df['sep_flux'] = flux
    obj_df['sep_fluxerr'] = fluxerr
    obj_df['sep_s/n'] = flux/fluxerr
    obj_df['sep_flag'] = flag

    # npix/tnpix: ratio of the full detected footprint (npix, at the SEP
    # extraction threshold) to the deblended "core" area (tnpix). A real
    # point source's footprint and core area are nearly the same thing
    # (ratio ~1.1-1.7, empirically, regardless of S/N or background level —
    # verified by injection testing in both the illuminated field and a
    # totally dark part of the detector). Cosmic rays consistently run much
    # higher (~2.2-3.6): even ones that pass the psf_like/elongation shape
    # checks tend to have irregular low-level structure (secondary tracks,
    # nearby associated hits from the same event) attached to an otherwise
    # compact core, inflating the total footprint without inflating the
    # core. This is a far more effective cosmic-ray discriminant than shape
    # correlation alone or lacosmic (which only catches ~10% of the cosmic
    # rays that pass the shape cuts, since lacosmic's sharp-edge assumption
    # doesn't hold at this detector's pixel scale) — a cutoff of 2.0 here
    # catches ~97% of known cosmic rays at only a ~2.5% false-reject cost on
    # real point sources.
    npix_ratio = obj_df['npix'] / obj_df['tnpix'].replace(0, np.nan)
    obj_df['npix_ratio'] = npix_ratio

    # apply filtering: edge exclusion + PSF shape (correlation + fit didn't hit bound) + size
    ny, nx = data.shape
    fit_dx = (obj_df['x_psf'] - obj_df['x']).abs()
    fit_dy = (obj_df['y_psf'] - obj_df['y']).abs()
    filter_mask = (obj_df['x'].between(edge_margin_x, nx - edge_margin_x) &
                   obj_df['y'].between(edge_margin_y, ny - edge_margin_y) &
                   (obj_df['psf_like'] >= psf_corr_thresh) &
                   (fit_dx < 0.9) & (fit_dy < 0.9) &
                   (npix_ratio < 2.0))
    if psf_fwhm is not None:
        sigma_exp = psf_fwhm / (2 * np.sqrt(2 * np.log(2)))
        # SEP's isophotal 'a' for a genuine (fixed-shape) PSF isn't brightness
        # independent: extraction uses a fixed absolute threshold, so a
        # brighter source's isophote reaches further into the same Gaussian
        # wings before dropping below it. For a 2D Gaussian with std
        # sigma_exp, the radius at which it crosses a given threshold is
        # sigma_exp*sqrt(2*ln(peak/thresh)) -- growing with brightness. A
        # brightness-independent cutoff (the old `2.0*sigma_exp`) therefore
        # increasingly rejects real, genuinely round bright point sources
        # (confirmed via injection-recovery testing: real S/N~200 sources
        # were rejected ~70% of the time purely from this effect). Compare
        # against this brightness-aware expectation instead, with a 1.5x
        # margin for real PSF wings being a bit fatter than an ideal Gaussian
        # and for ordinary SEP measurement noise.
        ratio = np.maximum(obj_df['peak'] / obj_df['thresh'], 1.0001)
        a_expected = sigma_exp * np.sqrt(2 * np.log(ratio))
        a_expected = np.maximum(a_expected, sigma_exp)  # floor: never tighter than the old low-S/N limit
        filter_mask = filter_mask & (obj_df['a'] < 1.5 * a_expected)

    if cr_mask is not None:
        n_frame_total = cr_mask.shape[0]
        f0, f1 = max(0, frame - cr_frame_window), min(n_frame_total, frame + cr_frame_window + 1)
        cr_window = cr_mask[f0:f1]
        is_cr = np.zeros(len(obj_df), dtype=bool)
        for i, (xi, yi) in enumerate(zip(obj_df['x'].values, obj_df['y'].values)):
            xi, yi = int(round(xi)), int(round(yi))
            y0, y1 = max(0, yi - cr_px_window), min(ny, yi + cr_px_window + 1)
            x0, x1 = max(0, xi - cr_px_window), min(nx, xi + cr_px_window + 1)
            is_cr[i] = cr_window[:, y0:y1, x0:x1].any()
        obj_df['cr_flagged'] = is_cr
        filter_mask = filter_mask & ~is_cr
    else:
        obj_df['cr_flagged'] = False

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


def _measure_fwhm_px(psf_array):
    """
    Measures FWHM (in pixels) of a detector-sampled PSF array by linearly
    interpolating the half-max crossings on either side of the peak, along
    the central row. Used for instruments without a pre-tabulated FWHM
    lookup (e.g. NIRCam, whose FWHM varies strongly across its ~30 filters
    and two pixel scales). Sub-pixel interpolation matters here: NIRCam's
    SW channel is undersampled enough that sometimes only a single detector
    pixel sits above half-max, which a whole-pixel-counting measurement
    can't resolve.
    """
    ny, nx = psf_array.shape
    row = psf_array[ny // 2]
    peak = int(np.argmax(row))
    half = row[peak] / 2.0

    left = peak
    while left > 0 and row[left] > half:
        left -= 1
    right = peak
    while right < len(row) - 1 and row[right] > half:
        right += 1
    if left == peak or right == peak:
        return np.nan

    x_left = left + (half - row[left]) / (row[left + 1] - row[left])
    x_right = (right - 1) + (row[right - 1] - half) / (row[right - 1] - row[right])
    return float(x_right - x_left)


def _forward_diff(cube):
    """
    Forward (not centered) difference along axis 0: out[i] = cube[i+1] - cube[i].

    np.gradient's default centered difference, (cube[i+1]-cube[i-1])/2, is
    the wrong tool for discrete up-the-ramp reads: a single-group event
    (e.g. a cosmic ray) shows up smeared across two adjacent output frames,
    since the centered stencil at index i straddles it from both sides
    whichever group it lands in. A forward difference keeps a single-group
    event in exactly one output frame. Keeps the same shape as the input by
    setting the last frame to NaN (there's no cube[i+1] for it) -- that
    index is always in Jurassic.bad_frames anyway (last frame of the last
    integration), so nothing downstream relies on it having a value.
    """
    out = np.full_like(cube, np.nan)
    out[:-1] = np.diff(cube, axis=0)
    return out


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
        Class for searching the ramps of full array MIRI/NIRCam images for fast transients

        JURASSIC: JWST Up the Ramp Analysis Searching the Sky for Infrared Transients
    """

    def __init__(self,file=None,num_cores=35,run=True,method='mega',ramps=None,images=True,
                 significance=False,mask_correction=True,plot=True,no_sat_mask=False,
                 base_dir=None,data_dir=None,correct_ramps=True):
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
        self.correct_ramps = correct_ramps
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
        self.stpsf_class = { # instrument name -> stpsf class name
            "MIRI": "MIRI",
            "NIRCAM": "NIRCam",
        }

        if run:
            self._assign_data()
            if self.correct_ramps and self.instrument != 'MIRI':
                print(f'{self.instrument}: BFE/RCD ramp correction is only validated for '
                      f'MIRI — skipping ramp_correction() and using uncorrected ramps.')
                self.correct_ramps = False
            if self.correct_ramps:
                self.ramp_correction(cube=self.data)
            else:
                self.data_cor = self.data
                ny, nx = self.data.shape[2], self.data.shape[3]
                self.gen_mask = self._get_gen_mask(ny, nx)
            self.flux_calibrate(cube=self.data_cor)
            self._make_cubes()
            del self.data, self.data_cor, self.flux_data
            self._mask_pixels()
            del self.rampy_cube_dn, self.dq_cube

            if ramps: # search on ramp level
                print('ramps')
                self.parallel_fit_df(self.rampy_cube) # only fitting rampy_cube not mega

            if self.method == 'mega':
                if images or significance:
                    print('images')
                    self.mega_inator(self.rampy_cube)
                    del self.rampy_cube
                    self._cube_gradient(self.mega_cube_masked, save=True)
                    del self.mega_cube, self.mega_cube_masked, self.fakey_cube
                    self._reference_frame()
                    self._cube_differenced(self.grad_cube, self.first_ref_frame, save=False, first=None)
                    self._psf_kernel()
                    self.source_extracting(self.diff_cube, save_plot=False, save_csv=True)

                    print('re-difference')
                    self._masked_reference(self.mask_correction)
                    self._cube_differenced(self.grad_cube, self.second_ref_frame, save=True)
                    del self.grad_cube
                    self._remove_cosmic(self.diff_cube)
                    del self.diff_cube
                    self._make_ref_cr_mask()
                    self.source_extracting(self.clean_cube, save_plot=True, save_csv=False)

                if significance:
                    print('significance')
                    self._cube_significance()
                    self._cube_threshold()
                    del self.sig_cube, self.conv_sig_cube
                    self._cube_rolling_sum()
                    del self.bool_threshold_cube
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

            if not hasattr(self, 'significance_df'):
                self.significance_df = pd.DataFrame()
            self._time_mjd()
            self._flux_calibrate()
            self.save_outputs()


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

        # Remove the filename suffix (instrument/detector token + '_ramp.fits',
        # e.g. '..._mirimage_ramp.fits' or '..._nrcb1_ramp.fits')
        suffix1 = '_ramp.fits'
        suffix2 = '_cal.fits'
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
        with fits.open(self.stage1_filepath, ignore_missing_end=True) as hdul:
            self.data = np.array(hdul[1].data)
            self.dq_2d_arr = np.array(hdul[2].data)
            try:
                self.dq_3d_arr = np.array(hdul[3].data)
            except (TypeError, OSError, ValueError):
                print('GROUPDQ truncated — using zero DQ array')
                self.dq_3d_arr = np.zeros(self.data.shape, dtype=np.uint8)
            phdr = hdul['PRIMARY'].header
            self.instrument = phdr['INSTRUME'].strip().upper()
            self.detector   = phdr.get('DETECTOR', '').strip().upper()
            self.pupil      = phdr.get('PUPIL', None)
            self.tgroup    = phdr['TGROUP']
            self.filename  = phdr.get('FILENAME', self.obs_id)
            self.filter    = phdr['FILTER']
            self.subarray  = phdr['SUBARRAY']
            self.targname  = phdr['TARGNAME']
            self.substrt1  = phdr.get('SUBSTRT1', 1)   # 1-indexed FITS column start
            self.substrt2  = phdr.get('SUBSTRT2', 1)   # 1-indexed FITS row start
            try:
                self.time_df = pd.DataFrame(hdul[7].data)
            except (IndexError, KeyError):
                n_i = len(self.data)
                effinttm_days = phdr.get('EFFINTTM', self.tgroup * phdr.get('NGROUPS', 1)) / 86400.0
                t0 = phdr.get('EXPSTART', 0.0)
                starts = t0 + np.arange(n_i) * effinttm_days
                self.time_df = pd.DataFrame({
                    'integration_number': np.arange(1, n_i + 1),
                    'int_start_MJD_UTC':  starts,
                    'int_mid_MJD_UTC':    starts + effinttm_days / 2,
                    'int_end_MJD_UTC':    starts + effinttm_days,
                    'int_start_BJD_TDB':  starts,
                    'int_mid_BJD_TDB':    starts + effinttm_days / 2,
                    'int_end_BJD_TDB':    starts + effinttm_days,
                })
        # assigning data from cal.fits file
        self.pixar_sr = None
        if self.do_flux_cal:
            with fits.open(self.stage2_filepath) as hdul:
                self.cal_data = hdul[1].data
                self.pixar_sr = hdul[1].header.get('PIXAR_SR', None)
            m, s = np.nanmedian(self.cal_data), np.nanstd(self.cal_data)
            plt.figure()
            plt.imshow(self.cal_data,origin='lower',vmin=m-s,vmax=m+s)
            plt.savefig(os.path.join(self.obs_dir, 'cal_image.png'), bbox_inches="tight")


        self.n_int = len(self.data) # number of integrations (ramps) in file
        self.n_group = len(self.data[0]) # number of groups per integration
        self.n_frame = self.n_int * self.n_group # number of frames in file
        self.frames = list(range(self.n_frame)) # list of all frame indices

        bad_frames = []
        for integration in list(range(self.n_int)):
            bad_frames.append(integration*self.n_group)
            bad_frames.append(((integration+1)*self.n_group)-1)
        self.bad_frames = bad_frames

        self.fwhm = self._get_psf_fwhm_px()


    def _make_stpsf_instrument(self):
        """
        Builds the stpsf instrument object for self.instrument/self.filter,
        setting the detector too where relevant (e.g. NIRCam, which has
        multiple SCAs with different pixel scales).
        """
        cls_name = self.stpsf_class.get(self.instrument)
        if cls_name is None:
            raise ValueError(f'No stpsf model mapping for instrument {self.instrument}')
        inst = getattr(stpsf, cls_name)()
        inst.filter = self.filter
        if self.instrument == 'NIRCAM' and self.detector:
            # FITS DETECTOR keyword uses NRCA/BLONG for the LW channel;
            # stpsf/webbpsf names that SCA NRCA/B5 instead.
            inst.detector = self.detector.replace('LONG', '5')
        return inst


    def _get_psf_fwhm_px(self):
        """
        Returns the PSF FWHM in pixels for self.filter. MIRI uses the
        pre-tabulated JDOX values (self.psf_fwhm_px); other instruments
        (e.g. NIRCam, whose FWHM varies strongly across ~30 filters and
        two pixel scales) measure it directly from an stpsf model instead
        of relying on a hand-maintained table.
        """
        if self.instrument == 'MIRI':
            if self.filter in self.psf_fwhm_px:
                return self.psf_fwhm_px[self.filter]
            raise ValueError(f'Unknown MIRI filter {self.filter}: no FWHM available')

        inst = self._make_stpsf_instrument()
        psf = inst.calc_psf(fov_pixels=41)
        fwhm = _measure_fwhm_px(psf[3].data)
        if np.isnan(fwhm):
            raise ValueError(f'Could not measure PSF FWHM for {self.instrument}/{self.filter}')
        return fwhm


    def _get_gen_mask(self, ny, nx):
        """
        Returns the (ny, nx) boolean science-pixel mask (True = good).
        MIRI uses the bundled full-frame bad-pixel mask, cropped to the
        subarray (SUBSTRT are 1-indexed FITS coords). Other instruments have
        no bundled mask, so fall back to a DQ-derived mask from the PIXELDQ
        extension (DO_NOT_USE bit, bit 0 of the JWST DQ flag scheme).
        """
        if self.instrument == 'MIRI':
            _mask_path = os.path.join(os.path.dirname(__file__), 'full_MIRI_mask.npy')
            full_mask = np.load(_mask_path)
            if full_mask.shape == (ny, nx):
                return full_mask
            r0 = self.substrt2 - 1
            c0 = self.substrt1 - 1
            return full_mask[r0:r0+ny, c0:c0+nx]

        return (self.dq_2d_arr & 1) == 0


    def ramp_correction(self,cube):
        """
        Uses rampdoctor to correct both the brighter-fatter effect
        and the reset switch charge decay effects.

        RCD (reset charge decay) is a physical effect specific to MIRI's
        Si:As detectors; NIRCam's HgCdTe detectors don't exhibit it, and BFE
        hasn't been separately characterized/validated for NIRCam with this
        tool yet, so this should only be called for MIRI (see __init__).
        """
        from rampdoctor import RampDoctor
        ny, nx = cube.shape[2], cube.shape[3]
        self.gen_mask = self._get_gen_mask(ny, nx)
        rd = RampDoctor(cube=cube,bg_mask=self.gen_mask,sci_mask=self.gen_mask,verbose=True)

        self.data_cor = rd.correct(diagnostics=True, charge_adaptive=False)


    def flux_calibrate(self,cube):
        """
        Calibrates the 4-dimensional ramp data (self.data)
        Using the information from the reference files, via the
        instrument-appropriate photom data model (e.g. MirImgPhotomModel,
        NrcImgPhotomModel).
        """
        photom_model_cls = { # instrument -> imaging PHOTOM datamodel class name
            'MIRI': 'MirImgPhotomModel',
            'NIRCAM': 'NrcImgPhotomModel',
        }

        photmjsr = None
        uncertainty = None
        filt = self.filter

        try:
            from jwst import datamodels
            from stpipe import crds_client

            with datamodels.open(self.stage1_filepath) as model:
                crds_params = model.get_crds_parameters()
                filt = model.meta.instrument.filter
                pupil = model.meta.instrument.pupil

            photom_file = crds_client.get_reference_file(crds_params, 'photom', 'jwst')
            print(f"Using PHOTOM ref: {photom_file}")

            model_cls_name = photom_model_cls.get(self.instrument)
            if model_cls_name is None:
                raise ValueError(f'No PHOTOM datamodel mapping for instrument {self.instrument}')

            with getattr(datamodels, model_cls_name)(photom_file) as phot:
                table = phot.phot_table
                row_mask = table['filter'] == filt
                if 'pupil' in table.dtype.names and pupil is not None:
                    row_mask = row_mask & (table['pupil'] == pupil)
                row = table[row_mask]
                photmjsr = float(row['photmjsr'][0])
                uncertainty = float(row['uncertainty'][0])
        except Exception as e:
            print(f"CRDS flux cal failed ({e})", end='')
            if self.instrument == 'MIRI':
                print(" — falling back to miri_photom.csv")
            else:
                print(f" — no bundled fallback table for {self.instrument}")
                raise

        if photmjsr is None:
            csv_path = os.path.join(os.path.dirname(__file__), 'miri_photom.csv')
            phot_df = pd.read_csv(csv_path)
            mask = (phot_df['filter'] == filt) & (phot_df['subarray'] == self.subarray)
            if not mask.any():
                mask = phot_df['filter'] == filt
            row = phot_df[mask].iloc[0]
            photmjsr = float(row['photmjsr'])
            uncertainty = float(row['uncertainty'])

        print(f"filter={filt}  PHOTMJSR={photmjsr:.4f} MJy/sr per DN/s  +/- {uncertainty:.4f}")
        self.flux_conv = photmjsr
        self.flux_uncert = uncertainty

        # Not applied here — self.rampy_cube stays in raw DN and feeds
        # mega_inator / _cube_gradient / clean_cube unscaled. Only the final
        # gradient image (self.clean_cube) is flux-calibrated, in
        # _flux_calibrate(), to avoid double-calibrating and to keep frame 0
        # of every integration a legitimate DN value for mega_inator's
        # zero-point/extrapolation arithmetic.
        self.flux_data = cube


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

        self.jump_cube = (self.dq_cube & 4) == 4
        

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

        MIRI uses the calibrated DN threshold above (validated on MIRI data).
        Other instruments have no such calibrated threshold, so they use the
        SATURATED DQ flag (bit 1) from calwebb_detector1's own saturation
        step instead.
        """
        # load general mask (bad pixels / non-science)
        mask = self.gen_mask

        # mask out saturated pixels
        if self.instrument == 'MIRI':
            mask_sat = self.rampy_cube_dn[-1] < threshold
        else:
            mask_sat = (self.dq_cube[-1] & 2) == 0  # SATURATED = bit 1
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
        ng = self.n_group
        ni = self.n_int
        mega_cube = np.zeros((self.n_frame, cube.shape[1], cube.shape[2]))

        # Integration 0: zero relative to its first frame
        mega_cube[:ng] = cube[:ng] - cube[0]

        # Subsequent integrations: extrapolate the ramp value at the boundary
        for i in range(1, ni):
            i0 = i * ng
            difference = mega_cube[i0-2] + 2*(mega_cube[i0-2] - mega_cube[i0-3])
            mega_cube[i0:i0+ng] = cube[i0:i0+ng] - cube[i0] + difference

        # mask first and last frame of each integration
        bad = [i * ng for i in range(ni)] + [(i+1) * ng - 1 for i in range(ni)]
        mega_cube_masked = mega_cube.copy()
        mega_cube_masked[bad] = np.nan

        self.mega_cube = mega_cube
        self.mega_cube_masked = mega_cube_masked


    def _cube_gradient(self,cube,save=None):
        """
        make a gradient cube with the fakey fake frames for mega method
        for ramp method just takes the gradient then masks out bad frames

        Uses a forward difference (grad[i] = cube[i+1] - cube[i]), not
        np.gradient's centered difference. np.gradient approximates a
        continuous derivative from samples — for discrete up-the-ramp reads
        it has a real cost: a single-group event (a cosmic ray, a genuine
        one-group jump) gets smeared across *two* output frames, since the
        centered stencil at index i pulls in both cube[i-1] and cube[i+1].
        That makes a single-frame event look like two consecutive frames of
        signal, which is misleading for both visual inspection and any
        multi-frame persistence reasoning. A forward difference keeps a
        single-group event in exactly one output frame. The last frame has
        no cube[i+1] to diff against and is set to NaN — this is already
        always in self.bad_frames (the last frame of the last integration),
        so nothing already relied on it having a value.
        """
        if self.method == 'mega':
            fakeified_cube = cube.copy()
            vals = np.arange(1, self.n_int) * self.n_group
            if len(vals) > 0:
                fakeified_cube[vals - 1] = 2*fakeified_cube[vals - 2] - fakeified_cube[vals - 3]
                fakeified_cube[vals]     = 3*fakeified_cube[vals - 2] - 2*fakeified_cube[vals - 3]
            # Fakeify the absolute edge frames so the forward difference at
            # frame n_frame-2 doesn't need an out-of-range cube[n_frame], and
            # so small n_group (<=4 with n_int=1) still leaves valid frames.
            if self.n_group >= 3:
                fakeified_cube[0]  = 2*fakeified_cube[1]  - fakeified_cube[2]
                fakeified_cube[-1] = 2*fakeified_cube[-2] - fakeified_cube[-3]
            self.fakey_cube = fakeified_cube
            self.grad_cube = _forward_diff(fakeified_cube)

        if self.method == 'ramp':
            grad_cube = _forward_diff(cube)
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


    def _psf_kernel(self):
        """
        creates kernel based on filter using stpsf; size scales with PSF FWHM
        """
        fwhm = self.fwhm
        size = max(11, int(round(6 * fwhm)) | 1)  # odd, at least 11, ~3 FWHM radius
        inst = self._make_stpsf_instrument()
        psf = inst.calc_psf(fov_pixels=size)
        self.kernel = psf[3].data

    
    def source_extracting(self,cube,save_plot,save_csv):
        """
        using source extractor (sep) instead of StarFinder
        """
        psf_fwhm = getattr(self, 'fwhm', None)
        # lacosmic's mask only, NOT jump_cube: JUMP_DET is a pure ramp-level
        # statistical discontinuity test with no way to distinguish a cosmic
        # ray from a real fast transient brightening -- using it here would
        # systematically reject genuine discoveries. lacosmic instead tests
        # spatial PSF-consistency (is this sharp/unresolved vs. properly
        # optics-broadened), which a real transient satisfies regardless of
        # its time behavior, so it doesn't have that failure mode (verified
        # by injection testing — see cosmic_ray_shapes/).
        cr_mask = getattr(self, 'cr_mask_cube', None)  # None on the first pass, before it exists
        tasks = (delayed(_run_sep)(frame, cube[frame], self.kernel, self.mask_tot, save_plot, self.obs_dir, self.n_group, psf_fwhm,
                                    cr_mask=cr_mask)
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


    def _masked_reference(self,mask_correction,mask_radius=10,max_gap_frames=20):
        """
        Makes a reference frame (median) but masks out any variable sources.
        Masks detected source positions and takes nanmedian of remaining pixels.

        Two failure modes let a source leak into its own "background"
        reference, discovered investigating a slow-moving (~0.02 px/frame)
        bright MIRI asteroid that left a real, ~10+ sigma positive trace in
        ref_frame_2 along its own track, which then self-subtracted into a
        spurious negative dip whenever a given frame's true flux fell below
        that contaminated reference:

        1. mask_radius=10 px is smaller than the PSF model's own assumed
           extent (jurassic's WebbPSF kernel stamp half-size, ~6xFWHM/2 —
           13 px for F1500W). Masking a circle smaller than the PSF
           template itself guarantees real wing flux lands just outside
           the mask on every frame, biasing the reference high right where
           the source sits. Fixed by flooring mask_radius at the kernel's
           own half-size when self.kernel is available.
        2. The mask only covers frames where SEP actually reported a
           detection that frame. A frame where the source's per-frame
           significance dips below threshold — including, self-reinforcingly,
           frames already suffering from this exact contamination — gets
           no mask at all, so its source-containing pixels flow straight
           into the median uncorrected. Fixed by running this table through
           the pipeline's own trajectory linker (_spatial_group +
           _tag_asteroids) and, only for frames inside an asteroid-tagged
           track's own span, filling gaps of up to max_gap_frames by linear
           interpolation along that specific track's fitted trajectory.
           Scoping the fill to a single linked, classified track (rather
           than interpolating blindly between whatever SEP found in the
           nearest bracketing frames) avoids bridging two unrelated sources
           if the field has more than one — a real risk this early in the
           pipeline, since this runs on the first-pass, per-frame catalog,
           before the final grouping/classification later in the pipeline.
        """
        if self.filtered_sep_df.empty:
            self.second_ref_frame = self.first_ref_frame.copy()
            return

        if mask_correction == False:
            self.second_ref_frame = self.first_ref_frame.copy()
            return

        kernel_obj = getattr(self, 'kernel', None)
        if kernel_obj is not None:
            mask_radius = max(mask_radius, kernel_obj.shape[0] // 2)

        # gap-fill positions: only a fallback for frames with NO real
        # detection at all, so there's no multi-source ambiguity to resolve
        # here -- the mean per detected frame is just the interpolation
        # endpoint, not a replacement for that frame's own (possibly
        # multi-row) mask below.
        #
        # Scoped to single linked tracks: run the pipeline's own DBSCAN
        # spatial linker + trajectory classifier on this frame's detections
        # (using 'frame' as a monotonic stand-in for 'mjd' -- _tag_asteroids
        # only needs relative spacing for the linear fit, and real mjd
        # isn't assigned yet at this point in the pipeline) so a gap only
        # gets bridged when both bracketing detections have already been
        # identified, by that same linking logic, as the same moving object.
        det = self.filtered_sep_df
        linked = self._spatial_group(det[['x', 'y', 'frame']].copy())
        linked['mjd'] = linked['frame'].astype(float)
        tagged = self._tag_asteroids(linked)

        gap_fill = {}  # frame -> (x, y), only for frames absent from filtered_sep_df
        for ast_id in sorted(tagged.loc[tagged['asteroid_id'] > 0, 'asteroid_id'].unique()):
            track = tagged[tagged['asteroid_id'] == ast_id].sort_values('frame')
            track_frames = track['frame'].values
            track_x = track['x'].values
            track_y = track['y'].values
            for f0, x0, y0, f1, x1, y1 in zip(track_frames[:-1], track_x[:-1], track_y[:-1],
                                               track_frames[1:], track_x[1:], track_y[1:]):
                gap = f1 - f0
                if 1 < gap <= max_gap_frames:
                    for f in range(f0 + 1, f1):
                        t = (f - f0) / gap
                        gap_fill[f] = (x0 + t * (x1 - x0), y0 + t * (y1 - y0))

        # source masks: real per-frame detections keep their original,
        # possibly-multi-source handling; frames with no detection at all
        # fall back to the interpolated position, so a transient
        # non-detection (including one caused by this same self-subtraction
        # effect) no longer leaves an unmasked window
        reference_cube = np.zeros_like(self.grad_cube)
        kernel = self._circle_app(mask_radius)

        for frame in self.frames:
            has_detection = frame in self.filtered_sep_df['frame'].values
            has_fill = frame in gap_fill
            if not (has_detection or has_fill):
                continue

            mask = np.zeros_like(self.grad_cube[0])

            if has_detection:
                frame_df = self.filtered_sep_df[self.filtered_sep_df['frame'] == frame]
                x_int = [round(x) for x in frame_df['x'].values]
                y_int = [round(y) for y in frame_df['y'].values]
            else:
                gx, gy = gap_fill[frame]
                x_int, y_int = [round(gx)], [round(gy)]

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

        # A pixel goes NaN here only if the source's mask covered it in
        # every single valid frame -- true whenever the source's own
        # trajectory over the segment is smaller than the mask footprint
        # (exactly this slow-moving asteroid: ~12-16 px of total drift
        # across the segment vs a ~26 px mask diameter), so there's no time
        # in the whole segment this pixel is ever clear. Left as NaN, that
        # hole poisons diff_cube = grad_cube - reference at that exact
        # detector position for every frame in the segment, not just some
        # -- silently erasing the source's own detectability across
        # whatever portion of the track sits deep enough in its own mask
        # footprint to starve every frame (this is what turned a modest,
        # ~10 sigma reference contamination into a ~280-frame dead zone
        # with zero detections, found testing on a single segment before
        # running the full 7-segment set). Fill any such holes from the
        # local background via Gaussian interpolation instead of leaving
        # them empty -- a locally-smooth fallback beats an outright hole,
        # even though it's a coarser estimate than a real temporal median.
        nan_holes = np.isnan(self.second_ref_frame)
        if nan_holes.any():
            from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
            filled = self.second_ref_frame.copy()
            stddev = mask_radius
            for _ in range(4):  # widen the kernel until every hole is bridged
                fill_kernel = Gaussian2DKernel(x_stddev=stddev)
                filled = interpolate_replace_nans(filled, fill_kernel)
                if not np.isnan(filled).any():
                    break
                stddev *= 2
            self.second_ref_frame = filled

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

        # run_lacosmic zeroes NaN input (lacosmic can't accept NaN) and never
        # restores it, so the first/last frame of every integration — masked
        # NaN upstream in diff_cube — would otherwise silently become 0 here.
        clean_cube[self.bad_frames] = np.nan

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
        dat = np.where(self.mask_tot[None, :, :], self.clean_cube, np.nan)

        def _frame_stats(frame_data):
            _, med, std = sigma_clipped_stats(frame_data)
            return med, std

        stats = Parallel(n_jobs=self.num_cores)(
            delayed(_frame_stats)(dat[f]) for f in range(self.n_frame))
        meds = np.array([s[0] for s in stats])
        stds = np.array([s[1] for s in stats])

        sig_cube = (dat - meds[:, None, None]) / stds[:, None, None]
        sig_cube[self.ref_cr_mask] = 0

        self.sig_cube = sig_cube
        self.bool_sig_cube = sig_cube > magic_number


    def _cube_threshold(self,rad=2,threshold=9):
        """
        convolves the significance cube with a circle and identifies
        the bits above a threshold, above which should be psf-like sources
        and below are cosmic ray junk stuffs (ideally)
        """
        kernel = self._circle_app(rad)
        results = Parallel(n_jobs=self.num_cores)(
            delayed(convolve_fft)(self.bool_sig_cube[f], kernel, normalize_kernel=False)
            for f in range(self.n_frame))
        conv_sig_cube = np.array(results)
        self.conv_sig_cube = conv_sig_cube
        self.bool_threshold_cube = conv_sig_cube > threshold


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
            end = df.loc[i, "int_end_MJD_UTC"]
            times.extend(np.linspace(start,end,self.n_group))

        data = {'frame': frames, 'mjd': times}

        self.frame_mjd_df = pd.DataFrame(data) 


    def assign_mjd(self,df):
        """
        for a given pd dataframe with a column 'frame' will assign a mjd column
        """
        df = df.merge(self.frame_mjd_df, on="frame", how="left")

        return df


    def _psf_correlation(self, df, cutout_half=5):
        """
        Compute Pearson correlation between each detection cutout and a 2D Gaussian
        PSF model (sigma = FWHM/2.355, sub-pixel shifted to the source centroid).
        Adds 'psf_like' column; 1.0 = perfect PSF match, lower = extended/noise/CR.
        """
        from scipy.ndimage import shift as nd_shift

        sigma = self.fwhm / (2 * np.sqrt(2 * np.log(2)))
        size = 2 * cutout_half + 1
        yg, xg = np.mgrid[-cutout_half:cutout_half + 1, -cutout_half:cutout_half + 1]
        psf_base = np.exp(-(xg**2 + yg**2) / (2 * sigma**2))
        psf_base /= psf_base.sum()

        cube = self.diff_cube if hasattr(self, 'diff_cube') else self.clean_cube
        ny, nx = cube.shape[1], cube.shape[2]
        n_frames = cube.shape[0]

        psf_like = np.full(len(df), np.nan)

        for i, (_, row) in enumerate(df.iterrows()):
            frame = int(row['frame'])
            cx, cy = row['x'], row['y']
            xi, yi = int(round(cx)), int(round(cy))

            y0, y1 = yi - cutout_half, yi + cutout_half + 1
            x0, x1 = xi - cutout_half, xi + cutout_half + 1

            if frame >= n_frames or y0 < 0 or y1 > ny or x0 < 0 or x1 > nx:
                continue

            cutout = cube[frame, y0:y1, x0:x1].copy()
            if cutout.shape != (size, size) or np.all(np.isnan(cutout)):
                continue

            # Shift PSF to sub-pixel centroid position
            psf = nd_shift(psf_base, (cy - yi, cx - xi), mode='constant', cval=0)
            psf_sum = psf.sum()
            if psf_sum > 0:
                psf /= psf_sum

            valid = ~np.isnan(cutout)
            if valid.sum() < 4:
                continue

            r = np.corrcoef(cutout[valid].flatten(), psf[valid].flatten())[0, 1]
            psf_like[i] = r

        df = df.copy()
        df['psf_like'] = psf_like
        return df


    def make_video(self, objid, save_path, half=50, fps=20, dpi=100):
        """
        Save an MP4 video of a 2*half x 2*half px cutout centered on objid.
        Color range is set from the brightest event frame.
        Event frames are highlighted with a red border and annotated in the title.
        Colorbar is matched to the image height.
        """
        import matplotlib.animation as animation
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        obj = self.events[self.events['objid'] == objid].sort_values('frame')
        if obj.empty:
            raise ValueError(f'objid {objid} not found in self.events')

        cx = int(round(obj['x'].mean()))
        cy = int(round(obj['y'].mean()))
        event_frames = set(obj['frame'].astype(int).tolist())

        x0 = max(0, cx - half)
        x1 = min(self.clean_cube.shape[2], cx + half)
        y0 = max(0, cy - half)
        y1 = min(self.clean_cube.shape[1], cy + half)
        cutout = self.clean_cube[:, y0:y1, x0:x1]

        # Skip NaN/zero frames
        valid_frames = [i for i in range(cutout.shape[0])
                        if not np.all(np.isnan(cutout[i])) and not np.all(cutout[i] == 0)]

        # Color range from the brightest event frame
        bright_frame = int(obj.loc[obj['sep_flux'].idxmax(), 'frame'])
        bright_data = cutout[bright_frame]
        vmin = np.nanpercentile(cutout[valid_frames], 1)
        vmax = np.nanpercentile(bright_data[~np.isnan(bright_data)], 99) if not np.all(np.isnan(bright_data)) else vmin + 1

        fig, ax = plt.subplots(figsize=(5, 5))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)

        im = ax.imshow(cutout[valid_frames[0]], origin='lower', cmap='gray', vmin=vmin, vmax=vmax, animated=True)
        plt.colorbar(im, cax=cax, label='DN/group')
        ax.axvline(cx - x0, color='r', lw=0.5, alpha=0.4)
        ax.axhline(cy - y0, color='r', lw=0.5, alpha=0.4)
        title = ax.set_title(f'Frame {valid_frames[0]}', fontsize=12)

        for spine in ax.spines.values():
            spine.set_linewidth(2)

        def update(i):
            frame_idx = valid_frames[i]
            im.set_data(cutout[frame_idx])
            is_event = frame_idx in event_frames
            color = 'red' if is_event else 'black'
            label = '  [EVENT]' if is_event else ''
            title.set_text(f'Frame {frame_idx}{label}')
            title.set_color(color)
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
            return im, title

        fig.tight_layout()
        ani = animation.FuncAnimation(fig, update, frames=len(valid_frames),
                                      interval=1000 / fps, blit=False)
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        ani.save(save_path, writer='ffmpeg', fps=fps, dpi=dpi)
        plt.close(fig)
        print(f'Video saved to {save_path}')


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

        calibrated = getattr(self, 'cube_units', None) == 'mjy/sr'
        has_pixar = calibrated and getattr(self, 'pixar_sr', None) is not None
        if calibrated:
            if lc_units == 'ujy' and has_pixar:
                _ylabel = r'$\mu$Jy'
            elif lc_units == 'mjy' and has_pixar:
                _ylabel = 'MJy'
            else:
                _ylabel = r'MJy sr$^{-1}$' if not latex else r'MJy\,sr$^{-1}$'
        else:
            _ylabel = 'DN/s' if lc_units == 'dn/s' else 'DN/group'

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
                if calibrated:
                    n_pix = np.sum(~np.isnan(aperture), axis=(1, 2)).astype(float)
                    n_pix[n_pix == 0] = np.nan
                    if lc_units == 'ujy' and has_pixar:
                        f = np.where(all_nan, np.nan, np.nansum(aperture, axis=(1, 2))) * self.pixar_sr * 1e6
                    elif lc_units == 'mjy' and has_pixar:
                        f = np.where(all_nan, np.nan, np.nansum(aperture, axis=(1, 2))) * self.pixar_sr
                    else:
                        f = np.where(all_nan, np.nan, np.nansum(aperture, axis=(1, 2)) / n_pix)
                else:
                    f = np.where(all_nan, np.nan, np.nansum(aperture, axis=(1, 2)))
                    if lc_units == 'dn/s':
                        f = f / group_time_s

                # Strip NaN frames from both time and f; recompute breaks on clean arrays
                valid_lc = ~np.isnan(f) & ~np.isnan(time)
                time_c = time[valid_lc]
                f_c = f[valid_lc]
                med_c = np.nanmedian(np.diff(time_c))
                std_c = np.nanstd(np.diff(time_c))
                brk = np.where(np.diff(time_c) > med_c + std_c)[0]
                brk = np.insert(np.append(brk + 1, len(time_c)), 0, 0)

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

                fig, ax = plt.subplot_mosaic([[1, 1, 1, 2, 2], [1, 1, 1, 3, 3]],
                                             figsize=(7 * 1.1, 5.5 * 1.1), constrained_layout=True)

                # Ghost plot to fix inset ylims
                zoom_valid = valid_lc[fstart:frame_end + 20]
                zoom_t = time[fstart:frame_end + 20][zoom_valid]
                zoom_f = f[fstart:frame_end + 20][zoom_valid]
                ax[1].plot(zoom_t, zoom_f, 'k', alpha=0)
                insert_ylims = ax[1].get_ylim()

                # Full light curve — NaN frames already removed in time_c/f_c
                for seg in range(len(brk) - 1):
                    ax[1].plot(time_c[brk[seg]:brk[seg + 1]],
                               f_c[brk[seg]:brk[seg + 1]], 'k', alpha=0.8)

                ylims = ax[1].get_ylim()
                ax[1].set_ylim(ylims[0], ylims[1] + abs(ylims[0] - ylims[1]))
                ax[1].set_xlim(np.min(time_c), np.max(time_c))
                ax[1].set_title(f'ObjID: {objid}', fontsize=15)
                ax[1].set_ylabel(_ylabel, fontsize=15, labelpad=10)
                ax[1].set_xlabel(f'Time (MJD - {np.round(mjd_arr[0], 3)})', fontsize=15)

                axins = ax[1].inset_axes([0.1, 0.55, 0.86, 0.43])
                axins.axvspan(time[frame_start] - cadence / 2,
                              time[frame_end] + cadence / 2, color='C1', alpha=0.4)
                for seg in range(len(brk) - 1):
                    axins.plot(time_c[brk[seg]:brk[seg + 1]],
                               f_c[brk[seg]:brk[seg + 1]], 'k', alpha=0.8, marker='.')

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


    def _flux_calibrate(self):
        """
        Convert self.clean_cube from DN/group to MJy/sr, then divide by the
        group time so the result is in MJy/sr (surface brightness per pixel).

        MIRI uses the bundled miri_photom.csv (derived from
        jwst_miri_photom_0230.fits) with its exponential + linear
        time-dependent sensitivity-loss correction applied at the median
        observation MJD. No such drift model is available for other
        instruments, so they use the static CRDS photmjsr from
        flux_calibrate() (set on self.flux_conv) with no time correction.

        Sets:
            self.photmjsr : effective conversion factor [MJy/sr per DN/s]
            self.cube_units : 'mjy/sr' (used by plot_detection for unit handling)
        """
        mjd_arr = self.frame_mjd_df['mjd'].values
        group_time_s = np.nanmedian(np.diff(mjd_arr)) * 86400

        if self.instrument == 'MIRI':
            mjd = np.nanmedian(mjd_arr)

            csv_path = os.path.join(os.path.dirname(__file__), 'miri_photom.csv')
            phot = pd.read_csv(csv_path)

            filt = self.filter.strip()
            sub = self.subarray.strip()
            row = phot[(phot['filter'] == filt) & (phot['subarray'] == sub)]
            if row.empty:
                row = phot[(phot['filter'] == filt) & (phot['subarray'] == 'FULL')]
            if row.empty:
                raise ValueError(f'No photom entry for {filt}/{sub} in miri_photom.csv')
            row = row.iloc[0]

            dt_days = mjd - row['t0']
            exp_corr = row['const'] + row['amplitude'] * np.exp(-dt_days / row['tau'])
            lin_corr = 1.0 + row['lossperyear'] * (dt_days / 365.25)
            self.photmjsr = row['photmjsr'] * exp_corr * lin_corr

            print(f'Flux calibration: filter={filt}, subarray={sub}, '
                  f'MJD={mjd:.3f}, photmjsr={self.photmjsr:.4f} MJy/sr per DN/s')
        else:
            self.photmjsr = self.flux_conv

            print(f'Flux calibration: filter={self.filter}, instrument={self.instrument}, '
                  f'photmjsr={self.photmjsr:.4f} MJy/sr per DN/s (no time-dependent correction)')

        self.clean_cube = self.clean_cube * (self.photmjsr / group_time_s)
        self.cube_units = 'mjy/sr'


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
            self.events = self._psf_correlation(self.events)
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