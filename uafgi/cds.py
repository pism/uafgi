import sys
import os
import pickle
import cdsapi
import collections
import glob
import csv
import itertools
import time

MAX_QUEUE=3
RESOURCE = 'reanalysis-era5-single-levels'
CDS_ROOT = os.path.join(os.environ['HOME'], 'data_sets', 'cds', RESOURCE)

# --------------------------------------------------------------------------------------------
Combo = collections.namedtuple('Combo', ('var', 'year', 'month'))

def combo_to_fname(combo):
    """Provides the fully qualified pathname where a file specified by a combo lives."""
    return os.path.join(CDS_ROOT, combo.var, 'ERA5_{}_hourly_{:04d}_{:02d}.nc'.format(*combo))

def combo_to_json(combo):
    """Provides the JSON dict used to query a file based on a combo."""
    # resource = 'reanalysis-era5-single-levels'
    json = {
        'product_type':'reanalysis',
        'format': 'netcdf',
        'variable': combo.var,
        'year': '{:04d}'.format(combo.year),
        'month': '{:02d}'.format(combo.month),
        'day': [
            '01','02','03',
            '04','05','06',
            '07','08','09',
            '10','11','12',
            '13','14','15',
            '16','17','18',
            '19','20','21',
            '22','23','24',
            '25','26','27',
            '28','29','30',
            '31'
        ],
        'time': [
            '00:00', '01:00', '02:00',
            '03:00', '04:00', '05:00',
            '06:00', '07:00', '08:00',
            '09:00', '10:00', '11:00',
            '12:00', '13:00', '14:00',
            '15:00', '16:00', '17:00',
            '18:00', '19:00', '20:00',
            '21:00', '22:00', '23:00',
        ],
    }
    return json

def json_to_combo(json):
    """Reconstructs a combo from the JSON request.
    NOTE: json_to_combo(combo_to_json(...)) must be the identity.
    """

    # Be suspicious of requests that don't look like they came from us
    if 'variable' not in json:
        return None

    if isinstance(json['variable'], list) \
        or isinstance(json['month'], list) \
        or (json['format'] != 'netcdf'):
        return None

    return Combo(json['variable'], int(json['year']), int(json['month']))

# --------------------------------------------------------------------------------------------
def unique_combos():
    """Reads the set of all combos from all files cds_downloads/*.csv"""
    comboss = list()
    for fname in glob.glob(os.path.join(CDS_ROOT+'.todownload', '*.csv')):
        with open(fname, 'r') as fin:
            reader = csv.reader(fin)
            header = next(reader)
            comboss.append([Combo(row[0], int(row[1]), int(row[2])) for row in reader if len(row)>0])
    uniq_combos = list(set(itertools.chain(*comboss)))
    return uniq_combos



def do_downloads():
    """
    """

    # Get list of combos we want to download; filter out combos that
    # have already been downloaded.
    combos = unique_combos()
#    combos = list(combo for combo in unique_combos() if not os.path.exists(combo_to_fname(combo)))
    combos_set = set(combos)

    # Nothing to do if we've downloaded everything!
    print('We have requested {} combos'.format(len(combos)))
    if len(combos) == 0:
        return

    # Open a CDS API client
    client = cdsapi.Client(wait_until_complete=False, delete=False)

    # Obtain current state of server: tasks[request_id] = {...JSON dict...}
    # Request list of tasks; same as the "Your requests" in the web app.    
    # https://cds.climate.copernicus.eu/modules/custom/cds_apikeys/app/apidocs.html
    tasks = dict()
    for task in client.robust(client.session.get)('{}/tasks'.format(client.url)).json():
        if 'request_id' not in task:
            print('Task missing request_id: {}'.format(task), file=sys.stderr)
            continue

        tasks[task['request_id']] = task

    to_download = list()
    nqueued = 0    # Number of downloads currently queued or running
    for task in tasks.values():

        # Back out original combo from this task
        details = client.robust(client.session.get)('{}/tasks/{}/provenance'.format(client.url, task['request_id'])).json()
        json = details['original_request']['specific_json']
        combo = json_to_combo(json)

        # Ignore requests that didn't come from us
        if combo is None:
            continue
        print('TASK: {} {}'.format(combo, task))
        print('  ---> combo in set: {}'.format(combo in combos_set))

        # Keep track of how many request are "in flight" so we don't
        # overload the server down below.
        if task['state'] in ('queued', 'running'):
            nqueued += 1

        # Nothing to do if we don't want to download this combo
        # (probably because it's already been downloaded)
        if combo not in combos_set:
            continue

        # Download if this combo is ready
        if task['state'] == 'completed':
            to_download.append((combo, task['location']))

    # Write script to download files that are now downloadable
    # Create bash script that runs ALL commands; and returns an error code if any of them failed
    # https://stackoverflow.com/questions/54247481/run-set-of-commands-and-return-error-code-if-any-failed
    Makefile = CDS_ROOT + '.mk'
    os.makedirs(os.path.split(Makefile)[0], exist_ok=True)
    print('Writing Makefile: {}'.format(Makefile))
    with open(Makefile, 'w') as fout:
        fnames = [combo_to_fname(combo) for combo,_ in to_download]
        fout.write('all: {}\n\n'.format(' '.join(fnames)))

        for (combo,url),fname in zip(to_download,fnames):
            # --create-dirs added in curl 7.10.3
            cmd = ['curl', '--create-dirs', '-L', url, '--output', fname + '.download']

            fout.write('{}:\n'.format(fname))
            fout.write('\t{}\n'.format(' '.join(cmd)))
            fout.write('\tmv {}.download {}\n'.format(fname, fname))
            fout.write('\n')

    # Request more files if any space left in our queue
    additional_downloads = max(MAX_QUEUE - nqueued, 0)
    for combo in combos[:additional_downloads]:
        print('---------- Queuing request for {}'.format(combo))
        json = combo_to_json(combo)
        url = '{}/resources/{}'.format(client.url, RESOURCE)
        print(url)
        print(json)
        client.robust(client.session.post)(url, json=json)


    # Run the Makefile
    cmd = ['make', '-f', Makefile]
    subprocess.run(cmd)


do_downloads()
