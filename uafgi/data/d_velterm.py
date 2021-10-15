import pandas as pd
import uafgi.data
import os,re


def read():
    """Reads a single dataframe for all the velocity/terminus combos"""
    veltermRE = re.compile(r'velterm_(\d\d\d)\.df')
    dir = uafgi.data.join_outputs('velterm')
    dfs = list()
    for leaf in sorted(os.listdir(dir)):
        match = veltermRE.match(leaf)
        if match is None:
            continue
        glacier_id = int(match.group(1))
        #print(leaf, glacier_id)
        fname = os.path.join(dir, leaf)
        df = pd.read_pickle(fname)
        df['glacier_id'] = glacier_id
        dfs.append(df)

    return pd.concat(dfs)

