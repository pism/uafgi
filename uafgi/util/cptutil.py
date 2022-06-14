import collections,re
import matplotlib

def hsv2rgb(h, s, v):
    """Convert HSV colors to RGB.
    Args:
        h (scalar int in range [0,255])
        s,v (scalar double in range [0,1])
    Returns:
        r, g b (scalar int in range [0,255]
    See:
        http://en.wikipedia.org/wiki/HSL_and_HSV
        http://code.activestate.com/recipes/576919-python-rgb-and-hsv-conversion/
    """

    # Implementation based on pseudo-code from Wikipedia.

    h = float(h)
    s = float(s)
    v = float(v)
    h60 = h / 60.0
    h60f = math.floor(h60)
    hi = int(h60f) % 6
    f = h60 - h60f
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r, g, b = 0, 0, 0
    if hi == 0: r, g, b = v, t, p
    elif hi == 1: r, g, b = q, v, p
    elif hi == 2: r, g, b = p, v, t
    elif hi == 3: r, g, b = p, q, v
    elif hi == 4: r, g, b = t, p, v
    elif hi == 5: r, g, b = v, p, q
    r, g, b = int(r * 255), int(g * 255), int(b * 255)
    return r, g, b
    
# ---------------------------------------------------------------
def rgb2hsv(r, g, b):
    """Convert RGB colors to HSV.
    Args:
        r, g b (scalar int in range [0,255]
    Returns:
        h (scalar int in range [0,255])
        s,v (scalar double in range [0,1])
    See:
        http://en.wikipedia.org/wiki/HSL_and_HSV
        http://code.activestate.com/recipes/576919-python-rgb-and-hsv-conversion/
    """
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r-g)/df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = df/mx
    v = mx
    return h, s, v
# ----------------------------------------------------------
CPTRet = collections.namedtuple('CPTRet', ('cmap', 'vmin', 'vmax'))

_color_modelRE = re.compile('#\s*COLOR_MODEL\s*=\s*(.*)')
_lineRE = re.compile('\s*([-+0123456789\.]\S*)\s+(\S*)\s+(\S*)\s+(\S*)\s+(\S*)\s+(\S*)\s+(\S*)\s+(\S*)')
def read_cpt(ifname, reverse=False) :
    """Parses an already-read cpt file.
    Args:
        cpt_str (string):
            Contents of a cpt file
        reverse (bool):
            If true, reverse the colors on this palette
    Returns:
        (see cpt())
    See:
        http://soliton.vm.bytemark.co.uk/pub/cpt-city/
        http://osdir.com/ml/python.matplotlib.general/2005-01/msg00023.html
        http://assorted-experience.blogspot.com/2007/07/custom-colormaps.html
    """

    # --------- Read the file
    cmap_vals = []
    cmap_rgbs = []
    use_hsv = False
    with open(ifname) as fin:
        for line in fin:
            match = _color_modelRE.match(line)
            if match is not None :
                smodel = match.group(1)
                if smodel == 'HSV' :
                    use_hsv = True
            else :
                match = _lineRE.match(line)
                if match is not None :
                    for base in [1, 5] :
                        val = match.group(base)
                        c1 = match.group(base+1)
                        c2 = match.group(base+2)
                        c3 = match.group(base+3)
#                       print 'base=' + str(base) + ', tuple=',val,c1,c2,c3, use_hsv
                        if use_hsv :
                            rgb = hsv2rgb(int(c1), float(c2), float(c3))
                            cmap_vals.append(float(val))
                            cmap_rgbs.append(rgb)
                        else :
                            cmap_vals.append(float(val))
                            cmap_rgbs.append((int(c1), int(c2), int(c3)))

    # Assemble into cmapx
    if reverse : cmap_rgbs.reverse()
    cmapx = list(zip(cmap_vals, cmap_rgbs))

    # ------------ Get the colormap's range
    vmin = cmapx[0][0]
    vmax = cmapx[-1][0]

    # ------------- Create the colormap, converting the form it's in
    vrange = vmax - vmin
    rgbs = ([],[],[])

    c0 = cmapx[0]
    cur_val = c0[0]
    cur_rgb = c0[1]
    for k in range(0,3) :
        rgbs[k].append(( (cur_val-vmin) / vrange, cur_rgb[k]/255.0, cur_rgb[k]/255.0))

    for i in range(1,len(cmapx)-1,2) :
        cur_rgb = cmapx[i][1]
        next_rgb = cmapx[i+1][1]
        cur_val = cmapx[i][0]   # also equals to next_val
        for k in range(0,3) :
            rgbs[k].append(( (cur_val-vmin)/vrange, cur_rgb[k]/255.0, next_rgb[k]/255.0))

    c0 = cmapx[-1]
    cur_val = c0[0]
    cur_rgb = c0[1]
    for k in range(0,3) :
        rgbs[k].append(( (cur_val-vmin)/vrange, cur_rgb[k]/255.0, cur_rgb[k]/255.0))

    cdict = {'red' : rgbs[0], 'green' : rgbs[1], 'blue' : rgbs[2]}
    cmap =  matplotlib.colors.LinearSegmentedColormap('my_colormap', cdict, 1024)

    return CPTRet(cmap, vmin, vmax)

