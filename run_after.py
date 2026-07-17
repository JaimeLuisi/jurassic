"""
run_after.py — run jurassic at the current dev HEAD (with rampdoctor).
Outputs saved to test_after/.
"""
import sys, os

os.environ.setdefault('WEBBPSF_PATH', '/Users/rridden/Documents/work/code/jwst/webbpsf-data')

try:
    import stpsf
except ModuleNotFoundError:
    import webbpsf
    sys.modules['stpsf'] = webbpsf

sys.path.insert(0, os.path.dirname(__file__))

import jurassic as jmod

RAMP = ('/Users/rridden/Documents/work/code/jwst/ramps/trappist/'
        'jw01177007001_03101_00001-seg001_mirimage_ramp.fits')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'test_after')

print(f'Running post-rampdoctor jurassic → {OUT_DIR}')
j = jmod.Jurassic(file=RAMP, method='mega', num_cores=8,
                  base_dir=OUT_DIR,
                  data_dir=os.path.dirname(RAMP))
print('Done.')
