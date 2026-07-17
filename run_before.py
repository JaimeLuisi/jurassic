"""
run_before.py — run jurassic at the pre-rampdoctor commit (f7a72b8).

Monkey-patches path handling so the hardcoded /home/phys/… paths are
replaced with portable equivalents, then runs the full pipeline.
Outputs saved to test_before/.
"""
import sys, os

os.environ.setdefault('WEBBPSF_PATH', '/Users/rridden/Documents/work/code/jwst/webbpsf-data')

# stpsf is the renamed webbpsf; stub it so both versions import cleanly
try:
    import stpsf
except ModuleNotFoundError:
    import webbpsf, unittest.mock
    sys.modules['stpsf'] = webbpsf

# Use the pre-rampdoctor worktree's jurassic, not the current dev version
BEFORE_DIR = os.path.join(os.path.dirname(__file__), '..', 'jurassic_before')
sys.path.insert(0, os.path.abspath(BEFORE_DIR))

# Patch full_MIRI_mask.npy path: the worktree expects it in cwd
os.chdir(os.path.abspath(BEFORE_DIR))

import jurassic as jmod

RAMP = ('/Users/rridden/Documents/work/code/jwst/ramps/trappist/'
        'jw01177007001_03101_00001-seg001_mirimage_ramp.fits')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'test_before')

# Patch _assign_data to use portable paths
_orig_assign = jmod.Jurassic._assign_data

def _patched_assign(self):
    # Override what the hardcoded __init__ set
    self.file = RAMP
    parts = self.file.replace('\\', '/').split('/')
    self.name = parts[-2] if len(parts) >= 2 else '.'
    self.obs_id = parts[-1]

    suffix1 = '_mirimage_ramp.fits'
    suffix2 = '_mirimage_cal.fits'
    obs_n = self.obs_id[:-len(suffix1)] if self.obs_id.endswith(suffix1) else self.obs_id
    obs_name = f"dr_{obs_n}"

    self.base_dir = OUT_DIR
    self.data_dir = os.path.dirname(RAMP)
    self.obs_dir  = os.path.join(self.base_dir, obs_name)
    os.makedirs(self.obs_dir, exist_ok=True)

    self.stage1_filepath = RAMP
    cal_path = os.path.join(self.data_dir, obs_n + suffix2)
    self.do_flux_cal = os.path.exists(cal_path)
    if self.do_flux_cal:
        self.stage2_filepath = cal_path
    else:
        print("No cal.fits found — skipping flux calibration")

    from astropy.io import fits
    import numpy as np
    import pandas as pd

    with fits.open(self.stage1_filepath, ignore_missing_end=True) as hdul:
        self.data      = np.array(hdul[1].data)
        self.dq_2d_arr = np.array(hdul[2].data)
        try:
            self.dq_3d_arr = np.array(hdul[3].data)
        except (TypeError, OSError, ValueError):
            print('GROUPDQ truncated — using zero DQ array')
            self.dq_3d_arr = np.zeros(self.data.shape, dtype=np.uint8)
        phdr           = hdul['PRIMARY'].header
        self.tgroup    = phdr['TGROUP']
        self.filter    = phdr['FILTER']
        self.subarray  = phdr['SUBARRAY']
        self.targname  = phdr['TARGNAME']
        self.filename  = phdr.get('FILENAME', self.obs_id)
        try:
            self.time_df = pd.DataFrame(hdul[7].data)
        except (IndexError, KeyError):
            n_i = len(self.data)
            effinttm_days = phdr.get('EFFINTTM', self.tgroup * phdr.get('NGROUPS', 1)) / 86400.0
            t0 = phdr.get('EXPSTART', 0.0)
            starts = np.arange(n_i) * effinttm_days + t0
            self.time_df = pd.DataFrame({
                'integration_number': np.arange(1, n_i + 1),
                'int_start_MJD_UTC':  starts,
                'int_mid_MJD_UTC':    starts + effinttm_days / 2,
                'int_end_MJD_UTC':    starts + effinttm_days,
                'int_start_BJD_TDB':  starts,
                'int_mid_BJD_TDB':    starts + effinttm_days / 2,
                'int_end_BJD_TDB':    starts + effinttm_days,
            })

    self.n_int   = len(self.data)
    self.n_group = len(self.data[0])
    self.n_frame = self.n_int * self.n_group
    self.frames  = list(range(self.n_frame))

# Also patch __init__ to fix the hardcoded file.removeprefix and name/obs_id split
_orig_init = jmod.Jurassic.__init__

def _patched_init(self, file=None, num_cores=35, run=True, method='mega',
                  ramps=None, images=True, significance=True,
                  mask_correction=True, plot=True):
    # portable path parsing instead of removeprefix + hardcoded split('/')
    self.file = file
    parts = (file or '').replace('\\', '/').split('/')
    self.name   = parts[-2] if len(parts) >= 2 else '.'
    self.obs_id = parts[-1]
    self.method = method
    self.plot   = plot
    self.mask_correction = mask_correction
    self.num_cores = num_cores
    self.psf_fwhm_px = {
        "F560W": 1.882, "F770W": 2.445, "F1000W": 2.982,
        "F1130W": 3.409, "F1280W": 3.818, "F1500W": 4.436,
        "F1800W": 5.373, "F2100W": 6.127, "F2550W": 7.300,
    }
    if run:
        self._assign_data()
        self.flux_calibrate(cube=self.data)
        self.correct_bfe_rcd(cube=self.flux_data)
        self._make_cubes()
        self._mask_pixels()
        if ramps:
            self.parallel_fit_df(self.rampy_cube)
        if self.method == 'mega':
            if images or significance:
                print('images')
                self.mega_inator(self.rampy_cube)
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
        self._time_mjd()
        self.save_outputs()
        if not hasattr(self, 'significance_df'):
            import pandas as pd
            self.significance_df = pd.DataFrame()

jmod.Jurassic.__init__    = _patched_init
jmod.Jurassic._assign_data = _patched_assign

print(f'Running pre-rampdoctor jurassic → {OUT_DIR}')
j = jmod.Jurassic(file=RAMP, method='mega', num_cores=8)
print('Done.')
