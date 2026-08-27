from __future__ import print_function
from cdo import Cdo
# from gdal import ogr, osr DEPRECATED
from osgeo import ogr, osr, gdal 
from getpass import getpass
from numpy import ndarray, asarray
from osgeo import gdal
from osgeo import ogr, gdal
from osgeo import ogr, osr
from osgeo import osr
from osgeo import osr, ogr, gdal
from setuptools import setup, find_packages
from uafgi import checksum, giutil
from uafgi import earthdata
from uafgi import ioutil
## from uafgi.functional import *  DEPRECATED; also, a *-import: NOT PYTHONIC
from uafgi.util import functional
from uafgi.util import cdoutil, ncutil, functional, gdalutil, ogrutil
from uafgi.util import functional, cfutil, gdalutil
from uafgi.util import functional, ogrutil, cfutil, ncutil, gisutil
from uafgi.util import gdalutil, osrutil, pdutil
from uafgi.util import gicollections
from uafgi.util import gicollections, giutil
from uafgi.util import gisutil
from uafgi.util import ioutil
from uafgi.util import ioutil, gicollections
from uafgi.util import ncutil, giutil
from uafgi.util import pathutil, gicollections
from uafgi.util import shputil
from uafgi.util.checksum import hashup
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import urlopen, Request, build_opener, HTTPCookieProcessor
# from urllib2 import urlopen, Request, HTTPError, URLError, build_opener, HTTPCookieProcessor - 
# DEPRECATED in Python3. Use urllib. See https://stackoverflow.com/questions/2792650/import-error-no-module-name-urllib2 
from urllib import HTTPError, URLError, build_opener, HTTPCookieProcessor, urlopen
# from urlparse import urlparse - DEPRECATED in Python3. Use urlparse
from urllib import urlparse 
import argparse
import base64
import bisect
import cartopy.crs
import cartopy.geodesic
import cf_units
import collections
import collections, re
import collections.abc
import contextlib
import copy
import datetime
# import dggs.data <-- only used for WRF, not our ERA5 environment 
import dill
import doctest
import filecmp
import functools
import getopt
import glob
import hashlib
import importlib
import inspect
import itertools
import json
import json, subprocess
import math
import matplotlib
import matplotlib.cm
import matplotlib.colors
import matplotlib.font_manager
import matplotlib.pyplot as plt
import netCDF4
import netCDF4, cf_units
import netrc
import numpy as np
import operator
import os
import os, pickle
import os, stat
import os.path
import pandas as pd
import pathlib
import pickle
import pyproj
import pytest
import re
import requests
import scipy.interpolate
import scipy.stats
import shapefile
import shapely
import shapely.geometry
import shapely.ops
import shutil
import signal
import ssl
import string
import struct
import subprocess
import sys
import tempfile
import time
import types
# import uafgi.indexing <-- this does not exist
import weakref
import xml.etree.ElementTree as ET