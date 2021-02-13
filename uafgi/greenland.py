import re
import pandas as pd
import itertools
# https://www.datacamp.com/community/tutorials/fuzzy-string-python
import Levenshtein

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


def fix_name(name):
    name = _replace_chars(name.lower())
    words = name.replace('_', ' ').split(' ')
    words = [_replace_words(w) for w in words]
    name = ' '.join(words)
    return name
 
def levenshtein(name0, name1):
    '''Determines if two names are the same or different'''
    n0 = fix_name(name0)
    n1 = fix_name(name1)
    ret = Levenshtein.ratio(n0,n1)
#    print('lev({}, {}) = {}'.format(n0,n1,ret))
    return ret
    
def max_levenshtein(names0, names1):
    """Computes maximum Levenshtein ratio of one name in names0 vs. one name in names1."""
    ratio = -1.
    for n0,n1 in itertools.product(names0,names1):
        ratio = max(ratio, levenshtein(n0,n1))
    return ratio

def levenshtein_cols(df0, df1):
    """df0 and df1 are DataFrames, selecting out just the cols we want to join.
    cols0 all share an index; cols1 all share a different index"""

    col_ix0 = list()
    col_ix1 = list()
    col_val = list()
    for ix0,row0 in df0.iterrows():
        row0_vals = [val for val in row0.values if len(val) > 0]
        for ix1,row1 in df1.iterrows():
            row1_vals = [val for val in row1.values if len(val) > 0]

            col_ix0.append(ix0)
            col_ix1.append(ix1)
            val = max_levenshtein(row0_vals, row1_vals)
            col_val.append(val)

#            if len(col_ix0) > 10:
#                break
#        if len(col_ix0) > 10:
#            break

    return pd.DataFrame({
        'ix0' : col_ix0,
        'ix1' : col_ix1,
        'val' : col_val,
    })


    
