import re,os,datetime,itertools
from uafgi import giutil,iopfile,gdalutil
import shapely
import pandas as pd

# Specifics of the data/ directory

# These are files for which domain_checksum2() == 0
blacklist_raw = (
    # ---------------------  W69.10N
    'TSX_W69.10N_02Jun18_13Jun18_09-48-58_{parameter}_v02.0{ext}',
    'TSX_W69.10N_29May15_20Jun15_09-48-37_{parameter}_v02.0{ext}',
    'TSX_W69.10N_31May14_11Jun14_09-48-32_{parameter}_v02.0{ext}',
    'TSX_W69.10N_03Jul09_14Jul09_09-48-07_{parameter}_v02.0{ext}',
    'TSX_W69.10N_13Jun18_05Jul18_09-48-58_{parameter}_v02.0{ext}',
    'TSX_W69.10N_05Jul18_27Jul18_09-49-00_{parameter}_v02.0{ext}',
    'TSX_W69.10N_26May16_17Jun16_09-48-43_{parameter}_v02.0{ext}',
    'TSX_W69.10N_18Jul17_09Aug17_09-48-53_{parameter}_v02.0{ext}',
    'TSX_W69.10N_16Jul13_18Aug13_09-48-29_{parameter}_v02.0{ext}',
    'TSX_W69.10N_30Apr18_02Jun18_09-48-57_{parameter}_v02.0{ext}',
    'TSX_W69.10N_10Nov15_21Nov15_09-48-42_{parameter}_v02.0{ext}',
    'TSX_W69.10N_23Aug11_14Sep11_09-48-21_{parameter}_v02.0{ext}',
    'TSX_W69.10N_18Sep09_29Sep09_09-48-11_{parameter}_v02.0{ext}',
    'TSX_W69.10N_28Apr09_09May09_09-48-04_{parameter}_v02.0{ext}',
    'TSX_W69.10N_09Sep18_20Sep18_09-49-03_{parameter}_v02.0{ext}',
    'TSX_W69.10N_30Jan09_10Feb09_09-48-02_{parameter}_v02.0{ext}',
    'TSX_W69.10N_06Feb11_28Feb11_09-48-12_{parameter}_v02.0{ext}',
    'TSX_W69.10N_21Apr12_02May12_09-48-19_{parameter}_v02.0{ext}',
    'TSX_W69.10N_10Feb14_21Feb14_09-48-29_{parameter}_v02.0{ext}',
    'TSX_W69.10N_23Nov14_04Dec14_09-48-46_{parameter}_v02.0{ext}',
    'TSX_W69.10N_25Aug10_05Sep10_09-48-16_{parameter}_v02.0{ext}',
    'TSX_W69.10N_21Nov10_02Dec10_09-48-17_{parameter}_v02.0{ext}',
    'TSX_W69.10N_14Feb17_25Feb17_09-48-47_{parameter}_v02.0{ext}',
    'TSX_W69.10N_18May15_29May15_09-48-36_{parameter}_v02.0{ext}',
    'TSX_W69.10N_04Feb12_15Feb12_09-48-17_{parameter}_v02.0{ext}',
    'TSX_W69.10N_19Apr18_30Apr18_09-48-57_{parameter}_v02.0{ext}',
    'TSX_W69.10N_21Jul11_01Aug11_09-48-19_{parameter}_v02.0{ext}',
    'TSX_W69.10N_12Feb13_23Feb13_09-48-22_{parameter}_v02.0{ext}',
    'TSX_W69.10N_24Apr11_05May11_09-48-14_{parameter}_v02.0{ext}',
    'TSX_W69.10N_26Apr10_07May10_09-48-11_{parameter}_v02.0{ext}',
    # Right domain but Missing a LOT
    'TSX_W69.10N_03Jul19_25Jul19_20-42-06_{parameter}_v02.0{ext}',
    'TSX_W69.10N_10Sep17_02Oct17_10-06-06_{parameter}_v02.0{ext}', 

    # ---------------------- W71.55N
)

def _get_blacklist(**kwargs):
    return [x.format(**kwargs) for x in blacklist_raw]

parameters = ('vx', 'vy')
blacklist = set(itertools.chain(*[_get_blacklist(parameter=x,ext='.tif') for x in parameters]))
# --------------------------------------------------------------------------------
class PFile(iopfile.PFile):

    key_fn = lambda x: (x['source'], x['grid'], x['startdate'], x['enddate'],
        x['parameter'], x['nominal_time'], x['version'], x['ext'])


    def format(self, **overrides):
        # Override self with overrides
        pfile = giutil.merge_dicts(self, overrides)

        pfile['sstartdate'] = datetime.datetime.strftime(pfile['startdate'], '%d%b%y')
        pfile['senddate'] = datetime.datetime.strftime(pfile['enddate'], '%d%b%y')
        pfile['snominal_time'] = '{:02d}-{:02d}-{:02d}'.format(*pfile['nominal_time'])
        # Override ext with user-given value
        if pfile['parameter'] == '':
            fmt = '{source}_{grid}_{sstartdate}_{senddate}_{snominal_time}_v{version}{ext}'
        else:
            fmt = '{source}_{grid}_{sstartdate}_{senddate}_{snominal_time}_{parameter}_v{version}{ext}'

        return fmt.format(**pfile)

imonth = { 'Jan':1, 'Feb':2, 'Mar':3, 'Apr':4, 'May':5, 'Jun':6, 'Jul':7,
    'Aug':8, 'Sep':9, 'Oct':10, 'Nov':11, 'Dec':12}


_reNSIDC = re.compile(r'(TSX|TDX)_([EWS][0-9.]+[NS])_(\d\d(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\d\d)_(\d\d(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\d\d)_(\d\d)-(\d\d)-(\d\d)(_(vv|vx|vy|ex|ey)?)_v([0-9.]+)(\..+)')

def parse(path):
    """
    See: https://org/data/nsidc-0481"""

    dir,leaf = os.path.split(path)

    match = _reNSIDC.match(leaf)
    if match is None:
        return None

    sstartdate = match.group(3)
    senddate = match.group(5)
    ret = PFile(
        dir=dir,
        leaf=leaf,
        source=match.group(1),
        grid=match.group(2),
        startdate=datetime.datetime.strptime(sstartdate, "%d%b%y"),
        enddate=datetime.datetime.strptime(senddate, "%d%b%y"),
        nominal_time=(int(match.group(7)), int(match.group(8)), int(match.group(9))),
        parameter=match.group(11),   # Could be None
        version=match.group(12),
        ext=match.group(13))

    if ret['parameter'] is None:
        ret['parameter'] = ''    # Don't like None for sorting

    return ret


def load_grids():

    """Loads all the grids in the NSIDC-0481 dataset, returning a
    rectangle for each domain.  Rectangle is in projected coordinates;
    note that all grids use the smae projection.
    poly:
        Shapely polygon of the grid bounding box
    srs:
        
    """
    griddir = os.path.join('data', 'measures', 'grids')
    gridRE = re.compile(r'([WE]\d\d\.\d\d[NS])_grid\.nc')

    grid_s = list()
    poly_s = list()
    wkt_s = list()
    for leaf in os.listdir(griddir):
        match = gridRE.match(leaf)
        if match is None:
            continue

        # Create a Shapely Polygon for the rectangle bounding box
        fi = gdalutil.FileInfo(os.path.join(griddir,leaf))
        grid_s.append(match.group(1))
        poly_s.append(shapely.geometry.Polygon([(fi.x0,fi.y0), (fi.x1,fi.y0), (fi.x1,fi.y1), (fi.x0,fi.y1)]))
        wkt_s.append(fi.srs.ExportToWkt())

    grids_df = pd.DataFrame({'grid': grid_s, 'poly': poly_s, 'wkt' : wkt_s})

    return grids_df
