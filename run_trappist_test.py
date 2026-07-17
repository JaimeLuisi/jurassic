"""
Test jurassic + rampdoctor on a TRAPPIST-1 ramp file (seg002, full array).
"""
import sys, os

os.environ.setdefault('WEBBPSF_PATH', '/Users/rridden/Documents/work/code/jwst/webbpsf-data')

sys.path.insert(0, os.path.dirname(__file__))

import jurassic as jmod

RAMP = ('/Users/rridden/Documents/work/code/jwst/ramps/trappist/'
        'jw01177007001_03101_00001-seg002_mirimage_ramp.fits')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'test_trappist')

print(f'Input:  {RAMP}')
print(f'Output: {OUT_DIR}')

j = jmod.Jurassic(file=RAMP, method='mega', num_cores=8,
                  base_dir=OUT_DIR,
                  data_dir=os.path.dirname(RAMP))
print('Done.')
