import csv
import pandas as pd
import itertools
from uafgi import ioutil

# Reads the dataset:
# data/GreenlandGlacierStats/
# 
# From the paper:
#
# Ocean forcing drives glacier retreat in Greenland
# Copyright © 2021
#   Michael Wood1,2*, Eric Rignot1,2, Ian Fenty2, Lu An1, Anders Bjørk3,
#   Michiel van den Broeke4, 11251 Cilan Cai , Emily Kane , Dimitris
#   Menemenlis , Romain Millan , Mathieu Morlighem , Jeremie
#   Mouginot1,5, Brice Noël4, Bernd Scheuchl1, Isabella Velicogna1,2,
#   Josh K. Willis2, Hong Zhang2

with ioutil.TmpDir() as tdir:
    # https://stackoverflow.com/questions/4869189/how-to-transpose-a-dataset-in-a-csv-file
    a = itertools.izip(*csv.reader(open("input.csv", "rb")))
    tfile = tdir.filename(suffix='.csv')
    with open(tfile, 'w') as fout:
        csv.writer(open(foud, "wb")).writerows(a)
    df = pd.read_csv(tfile)
print(df)
