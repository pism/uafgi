import datetime

from datetime import datetime as dt
import time

# https://www.py4u.net/discuss/142432
def year_fraction(date):
    def sinceEpoch(date): # returns seconds since epoch
        return time.mktime(date.timetuple())
    s = sinceEpoch

    year = date.year
    startOfThisYear = dt(year=year, month=1, day=1)
    startOfNextYear = dt(year=year+1, month=1, day=1)

    yearElapsed = s(date) - s(startOfThisYear)
    yearDuration = s(startOfNextYear) - s(startOfThisYear)
    fraction = yearElapsed/yearDuration

    return date.year + fraction


#def year_fraction(date):
#    start = datetime.date(date.year, 1, 1).toordinal()
#    year_length = datetime.date(date.year+1, 1, 1).toordinal() - start
#    return date.year + float(date.toordinal() - start) / year_lenght
