import re
import pandas as pd
import itertools
# https://www.datacamp.com/community/tutorials/fuzzy-string-python
import Levenshtein
from uafgi import pdutil

def sep_colnames(lst):
    """Process colnames and units into names list and units dict"""

    colnames = list()
    units = dict()
    for x in lst:
        if isinstance(x,tuple):
            colnames.append(x[0])
            units[x[0]] = x[1]
        else:
            colnames.append(x)
    return colnames, units



# ================================================================
# ================================================================
# ================================================================
# ================================================================

# ================================================================
# ================================================================
# ================================================================

# Good article on Greenland place names
# https://forum.arctic-sea-ice.net/index.php?topic=277.20;wap2

# https://stackoverflow.com/questions/6116978/how-to-replace-multiple-substrings-of-a-string
class ReplaceMany:
    def __init__(self, replaces):
        # use these three lines to do the replacement
        self.rep = dict((re.escape(k), v) for k, v in replaces) 
        self.pattern = re.compile("|".join(self.rep.keys()))

    def __call__(self, text):
        return self.pattern.sub(lambda m: self.rep[re.escape(m.group(0))], text)

_replace_chars = ReplaceMany([
    ('.', ''),
    ('æ', 'ae'),
#    ('Æ', 'AE'),
    ('ø', 'o'),
#    ('Ø', 'O'),
    ('â', 'a'),
    ('â', 'a'),
    ('ù', 'u'),
])

class ReplaceWords:
    def __init__(self, replaces):
        self.replaces = dict(replaces)
    def __call__(self, word):
        try:
            return self.replaces[word]
        except KeyError:
            return word

_replace_words = ReplaceWords([
    # "Glacier"
    ('gletsjer', 'gl'),
    ('gletscher', 'gl'),
    ('glacier', 'gl'),

    # "Glacier"
    ('se', 'sermia'),
#    ('sermia', 'se'),
#    ('sermiat', 'se'),
#    ('sermeq', 'se'),
#    ('sermilik', 'se'),
])


_dirRE = re.compile(r'^(n|s|e|w|nw|ss|c|sss|cs|cn)$')

def fix_name(name):
    name = _replace_chars(name.lower())
    words = name.replace('_', ' ').split(' ')
    words = [_replace_words(w) for w in words]


    # See if the last word is a direction designator.  Eg:
    #     Upernavik Isstrom N
    #     Upernavik Isstrom C
    #     Upernavik Isstrom S
    #     Upernavik Isstrom SS	
    match = _dirRE.match(words[-1])
    if match is not None:
        ret = (' '.join(words[:-1]), words[-1])
    else:
        ret = (' '.join(words), '')
    return ret
 
def levenshtein(name0, name1):
    '''Determines if two names are the same or different'''
    n0,dir0 = fix_name(name0)
    n1,dir1 = fix_name(name1)

    return (dir0==dir1, Levenshtein.ratio(n0,n1))
    
def max_levenshtein(names0, names1):
    """Computes maximum Levenshtein ratio of one name in names0 vs. one name in names1."""
    dir_and_ratio = (False,-1.)

#    nm0 = [x for x in names0 if len(x)>0]
#    nm1 = [x for x in names1 if len(x)>0]
    for n0,n1 in itertools.product(
        (x for x in names0 if type(x) != float and len(x)>0),
        (x for x in names1 if type(x) != float and len(x)>0)):
        dir_and_ratio = max(dir_and_ratio, levenshtein(n0,n1))

        # OPTIMIZATION: If we already found a full match, we're done!
        if dir_and_ratio == (True,1.0):
            return dir_and_ratio
    return dir_and_ratio

def levenshtein_cols(allnames0, allnames1):

    """For each row in a bunch of columns, compute the maximum Levenshtein
    ratio between any column in df0 and any column in df1.  Thus, it is likely
    to find a match if *any* column of df0 matches *any* column of df1.

    df0, df1: Dframe
        DataFrames consisting of columns to be joined.
        NOTE: This should ONLY contain columns to be joined
    Returns: DataFrame

df0 and df1 are DataFrames, selecting out just the cols we want to join.
    cols0 all share an index; cols1 all share a different index

    """

    col_ix0 = list()
    col_ix1 = list()
    col_dir = list()
    col_lev = list()
    for ix0,an0 in allnames0.iteritems():
        for ix1,an1 in allnames1.iteritems():

            col_ix0.append(ix0)
            col_ix1.append(ix1)
            dir,lev = max_levenshtein(an0, an1)
            col_dir.append(dir)
            col_lev.append(lev)

#            if len(col_ix0) > 10:
#                break
#        if len(col_ix0) > 1:
#            break

    return pd.DataFrame({
        'ix0' : col_ix0,
        'ix1' : col_ix1,
        'dir' : col_dir,
        'lev' : col_lev,
    })

# ============================================================



def match_allnames(xf0, xf1):

    """Finds candidate Matches between two DataFrames based on the
    *allnames* field.

    xf0, xf1: ExtDf
    """

    lcols = levenshtein_cols(
        xf0.df[xf0.prefix+'allnames'],
        xf1.df[xf1.prefix+'allnames']) \
        .rename(columns={'ix0':xf0.prefix+'ix', 'ix1':xf1.prefix+'ix'})

    # Only keep exact matches.  Problem is... inexact matches are fooled by
    # "XGlacier N", "XGlacier E", etc.
    lcols = lcols[lcols['lev']==1.0]

    dfx0 = xf0.df.loc[lcols[xf0.prefix+'ix']]
    dfx0.index = lcols.index

    dfx1 = xf1.df.loc[lcols[xf1.prefix+'ix']]
    dfx1.index = lcols.index

    xcols = lcols.copy()
    xcols[xf0.prefix+'key'] = dfx0[xf0.prefix+'key']
    xcols[xf1.prefix+'key'] = dfx1[xf1.prefix+'key']
    xcols[xf0.prefix+'allnames'] = dfx0[xf0.prefix+'allnames']
    xcols[xf1.prefix+'allnames'] = dfx1[xf1.prefix+'allnames']

    return pdutil.Match((xf0,xf1), (xf0.prefix+'allnames', xf1.prefix+'allnames'), xcols)

# -------------------------------------------------------------------
