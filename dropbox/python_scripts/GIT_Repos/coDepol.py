"""
[coDepol] : Compare the depolarization ratios from MPL and CORALNet datasets.

Creates (1) scatter plots of depolarization ratio for corresponding data sets, 
and (2) space-time difference plots. The scatter plots can be divided by 
altitude, and also on log axes.

Dependencies/built on:
----------------------
-- **build (64-bit)
-- python v 2.7.3
-- numpy v 1.6.1
-- matplotlib v 1.2.0

"""

from __future__ import print_function
import sys
import os
import math
import copy
import warnings
import operator
import numpy as np
import pandas as pd
import datetime as dt
from scipy.stats import linregress

from matplotlib import use as mpltuse
mpltuse('Agg')
from matplotlib import rcParams
rcParams.update({'font.family': 'serif', 'font.size': 16})

import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

__author__ = 'Annie Seagram'
__email__ = 'aseagram@eos.ubc.ca'
__created__ = 'September 27 2013'
__modified__ = 'October 27 2013'
__version__ = '1.3'

lib = 'C:\\Users\\dashamstyr\\Dropbox\\Python_Scripts'
try:
    sys.path.append(os.path.join(lib, 'LNCcode'))
    sys.path.append(os.path.join(lib, 'MPLcode'))
    from LNCcode import LNC_tools as lnctools
    from LNCcode import LNC_plot2 as lplot
    from MPLcode import MPL_plot as mplot
    from MPLcode import MPLtools as mtools
except ImportError:
    raise Exception('You havent specified where your modules are located!')
    


def filter_depol(depol, f=None):
    """Filter the depolarization data. The default filtering is from 0-1.
    
    Note: For use with pandas.DataFrames, be sure to nest the filter function
    calls (see example [2])
    
    Parameters
    ---------
    depol : array-like
        The depolarization data.
    f : tuple {optional}
        A rule to apply to the data as a tuple in the form (condition, value).
        The `condition` is an operator as a string ('gt', 'lt', etc), and the
        `value` is a number (int or float).
    
    Returns
    -------
    depol : 
        The filtered depolarization data.
    
    Example
    -------
    [1] >>> filter_depol(data, f=('le', 0.5))
    [2] >>> filter_depol(filter_depol(data, f=('le', 0.5)), f=('gt', 0.025))
    
    """
    
    _operator_map = {'gt': operator.gt, 'le': operator.le, 'eq': operator.eq,
                     'lt': operator.lt, 'ge': operator.ge, 'ne': operator.ne}
                     
    if not f:
        return depol[(depol > 0) & (depol <= 1)]
    else:
        value = f[-1]
        condition = _operator_map[f[0]]
        return depol[condition(depol, value)]
                        
def get_magnitude(x):
    """Find the magnitude of a number."""
    return math.floor(math.log10(x))

def linear_rvalue(x, y):
    """Returns the r_value from a linear regression between x and y.
    Note: This is NOT the r^2 value - The answer must be squared.
    
    """
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    return r_value

def _find_min(x, y):
    """Find the overall minimum of x and y data sets. Handles nan."""
    xf = np.ma.masked_where(np.isnan(x), x)
    yf = np.ma.masked_where(np.isnan(y), y)
    return np.nanmin([yf.min(), xf.min()])

def _custom_cmap(cmap, bounds):
    """Create a custom colormap for the depolarization plots. The colormap will
    be divided into discrete colors defined at the `bounds`.
    
    Parameters
    ----------
    cmap : str
        The name of a matplotlib cmap as a string (e.g. 'spectral')
    bounds: 
        The values of where to divide the cmap.
    
    Returns
    -------
    cmap : :class:~matplotlib.colors.LinearSegmentedColormap`
        The new discretized (linear segmented) cmap object.
    norm : :class:`~matplotlib.colors.BoundaryNorm`
        The normalizing object which scales data.
        
    """
    
    cmaplist = [cmap(i) for i in xrange(cmap.N)]
    
    # force the first color entry to be blue
    cmaplist[0] = (0, 0, 1, 1.0)
    # force the last color entry to be red
    cmaplist[-1] = (1, 0, 0, 1.0)
    
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
    cmap.set_over(color='r')

    # Normalize the map based on the bounds given
    norm = BoundaryNorm(bounds, cmap.N)
    
    return cmap, norm

def depol_scatter(x, y, s=15, cmap='gray_r', c='k', altbins=None, log=True, raster=True, grid=True, altunit='m', drawline=True, ax=None):
    """Create a depolarization scatter plot, that has linear or log-log axes.
    
    Values at different altitudes can be coloured based on *cmap*, *c*, and 
    *altbins*. See below for details.
    
    Note: rasterization is encouraged since usually many points will be 
    plotted (default True).
    
    Parameters
    ----------
    x, y : list, array like
        The data to plot on the scatter plot.
    s : int, default 15
        Size to render the points^2.
    cmap : str
        A valid matplotlib colormap as a string.
    c : str, tuple, array like
        - if c is a string or tuple, c is the color specification for the 
        points.
        - if c is array like, it must be the same shape as x that defines 
        the "z" values of the x,y points. These value will be mapped to the 
        *cmap*.
    altbins : list, array
        A sequence of altitudes that defines the bins to discretize the *cmap*.
        This only applies if *c* is provided as a list/array.
    log : bool, default True
        If *True*, the plot will be on log-log axes. If *False*, the plot will
        be on linear axes.
    raster : bool, default True
        Whether to rasterize the points on the plot. Recommended for large 
        numbers of points.
    grid : bool, default True
        If *True*, a gridlines will be plotted.
    altunit: ['m' | 'km']
        The scale for the altitude colorbar in meters ('m') or kilometers ('km').  
        Only used when *c* is implemented for altitude values. Default 'm'. 
    drawline : bool, default True.
        If *True*, draws a grey 1-to-1 line diagonally across the plot axes.
    ax : :class:`~matplotlib.axes.Axes`
        Axes instance of where to plot the data. Default None.
    
    Returns
    -------
    ax : :class:'~matplotlib.axes.Axes'
        The axes instance.
        
    """
    
    # Define a boolean for if the user supplied an ax kwarg
    # Note: in Python > 3.0, this can be changed to ax.__bool__
    givenAxes = ax is not None
    norm = None
    lowerLimit = 0
    axesMax = 1
    altbins = copy.copy(altbins)
    
    if not ax:
        # If no axes specified, make an axis on the current figure
        ax = plt.subplot(111)
    fig = ax.figure
    ax.set_rasterized(raster)
    
    if not givenAxes:
        fig.set_figheight(10)
        fig.set_figwidth(10)
    
    if not isinstance(c, (str, tuple)):
        
        # Adjust axes location for colorbar
        fig.subplots_adjust(bottom=0.075, left=0.085, right=0.85, top=0.925)
        
        # Adjust and normalize the colormap
        cmap = plt.cm.get_cmap(cmap)
        if altbins != None:
            extendAltBins = max(altbins) + 1
            if hasattr(altbins, 'append'):
                altbins.append(extendAltBins)
            else:
                altbins = np.array(altbins, extendAltBins)
            cmap, norm = _custom_cmap(cmap, altbins)
        else:
            pass
        
        # Add the colorbar
        divider = make_axes_locatable(ax)
        cbax = divider.append_axes('right', size='5%', pad=0.2)
        cb = ColorbarBase(cbax, cmap=cmap, norm=norm, boundaries=altbins, 
                         ticks=altbins, spacing='proportional', extend='max')
        
        if altunit == 'km':
            newlabels = []
            for label in cb.ax.get_yticklabels():
                value = label.get_text()
                newlabels.append('%.2f' % (float(value)/1000.))
            cb.ax.set_yticklabels(newlabels)
        cb.ax.set_ylabel('Altitude [%s]' % altunit, rotation=270)
        
    else:
        fig.subplots_adjust(bottom=0.1, left=0.075, right=0.925, top=0.925)
    
    if log:
        # Specifications for a log-log plot.
        #
        # This plot will fail if any value is zero.
        # Replace the 0 values with np.nan if present by using the .replace()
        # method. This is important since a boolean filter to replace 0 with 
        # nan(x[x>0]) will not alter the shape of a pandas.DataFrame, but it
        # WILL alter the shape of a pandas.Series, which will not be plotable.
        x = x.replace(0, np.nan)
        y = y.replace(0, np.nan)
        
        # To set the limits nicely, we can't use 0 as the min for a log-log 
        # plot. Instead, determine the order of magnitude of the lowest value
        # of the entire dataset (x,y) that is positive:
        minvalue = _find_min(x, y)
        if np.isfinite(minvalue):
            limitMagnitude = get_magnitude(minvalue)
            if limitMagnitude == 0:
                lowerLimit = 10**-1
            else:
                lowerLimit = 10**limitMagnitude
        else:
            lowerLimit = 10**-1
    
        if drawline:
            # Plot a 1-to-1 line
            ax.plot([lowerLimit, 1], [lowerLimit, 1], linestyle='-', lw=1,
                     color='0.5')    
        ax.set_xscale('log')
        ax.set_yscale('log')
    
    if not log:
        # Plot a 1-to-1 line based on the data present
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        axesMax = max([xmax, ymax])
        lowerLimit = min(xmin, xmax)
        if drawline:
            ax.plot([lowerLimit, axesMax], [lowerLimit, axesMax], 
                    linestyle='-', lw=1, color='0.5'
                    )
                    
    if grid:
        for val in np.arange(0.1, 1, 0.1):
            ax.axvline(x=val, linestyle=':', color='0.5')
            ax.axhline(y=val, linestyle=':', color='0.5')
            
    ax.scatter(x, y, s=s, c=c, cmap=cmap, norm=norm, edgecolors='None')
    
    ax.set_xlim(lowerLimit, axesMax)
    ax.set_ylim(lowerLimit, axesMax)
    ax.set_aspect(1)

    return ax

def depol_diff(x, y, xvals, yvals, vlim=0.1, cmap='RdBu_r', badvalue='0.25', ax=None):
    """Plot the difference between the two depolarization datasets. This 
    results in a space/time difference plot when x and y contain many profiles.
    
    Note: since this is a difference plot, the value of zero should be "forced"
    to the center of the colormap. This is why there is only one value *vlim*
    (rather than vmin and vmax), and only diverging colormaps are allowed.
    
    Parameters
    ----------
    x, y : array like, both (m,n)
        The x and y data sets to difference. The difference will be calculated
        as (y - x).
    xvals : list, array like (m,)
        The values for the x axis (e.g. time)
    yvals : list, array like (n,)
        The values for the y axis (e.g. altitude)
    vlim : int, float, default 0.1
        Limits of the difference data to display, sets the limits of the
        colormap. Note that vlim defines vmin = -1*vmax, which forces the
        value of 0 to the colormap.
    cmap : str
        A valid matplotlib DIVERGING colormap.
    badvalue : str, tuple
        A valid matlotlib color to set the color of np.nan values. If 
        *badvalue* is set to white, then a UserWarning is raised since 
        np.nan is now indistinguishable from zero. However, the user
        may want to view this for clarity.
    
    Returns
    -------
    ax : :class:`~matplotlib.axes.Axes`
        The axes instance.
    
    """
    
    cmapsDiverging = ['BrBG', 'bwr', 'coolwarm', 'PiYG', 'PRGn', 'PuOr',
                             'RdBu', 'RdGy', 'RdYlBu', 'RdYlGn', 'seismic']
    if (cmap not in cmapsDiverging) and (cmap not in [name+'_r' for name in cmapsDiverging]):
        raise ValueError('That is not a diverging colormap!')
    
    if not ax:
        # If no axes specified, make an axis on the current figure
        ax = plt.subplot(111)
    fig = ax.figure
    fig.subplots_adjust(bottom=0.1, left=0.1, right=.85, top=.95)
    
    # Difference:
    diff = y - x
    
    cmap = plt.cm.get_cmap(cmap)
    # To determine the difference between 0 and nan, set the bad color to 
    # non-white
    cmap.set_bad(color=badvalue)
    
    im = ax.imshow(diff[::-1], vmin=-vlim, vmax=vlim, cmap=cmap, rasterized=True, aspect='auto')
    ax.set_ylabel('Altitude [km]')
    ax.set_xlabel('Time [local]')
    # Note: your altticks label maker will round the km values
    mplot.altticks(ax, yvals[::-1]/1000., fsize=12)
    mplot.dateticks(ax, xvals, hours=['%02d' % hr for hr in range(2, 24, 2)], fsize=14)
    cb = fig.colorbar(im, extend='both')
    cb.ax.set_ylabel('Depolarization diff.', rotation=270)
    inc = (vlim+vlim)*0.25
    cb.set_ticks(np.arange(-vlim, vlim+inc, inc))
    
    if badvalue in ['1', '1.0', 'w', 'white', (0, 0, 0, 0)]:
        warnings.warn('badvalue {} : nan = zero = white'.format(badvalue))
        ax.set_title('<UNOFFICIAL> !! WARNING !! nan and zero = white')
        
    fig.tight_layout()
    
    return ax


def depol_scatter_byalt(x, y, altbins, xlabel, ylabel, s=10, c='k', log=True, raster=True, grid=True, suptitle=None, drawline=True, columns=4):
    """Create a depolarization scatter plot, that has linear or log-log axes,
    with subplots to divide the data set by altitude ranges.
    
    Note: 
        - Requires the xlabel and ylabel since this is displayed based on if 
        the axes is an outer axes of the subplot configuration.
        - Must supply *altbins* so that it knows how to divide the dataset.
        - *altbins* will automatically find the min and max of the dataset, 
        even if not specified. This assumes that any altitude range that is
        not desired is filtered out BEFORE the function call. E.g:
            if altbins = [2500] and the data ranges in altitude from 150 
            to 10000, then the bins plotted will be [150, 2500) and 
            [2500, 10000]. 
        This also means that if altbins contains a value that is out of the 
        range of the data, the value will be ignored. E.g.
            if altbins = [0, 250, 3000, 15000] and the data ranges from 150
            to 10000, then the bins plotted will be [150, 250), [250, 3000),
            [3000, 10000].

    Note: rasterization is encouraged since usually many points will be 
    plotted (default True).

    Parameters
    ----------
    x, y : :class:`~pandas.core.frame.DataFrame`
        The data to plot on the scatter plot as a pandas DataFrame
    altbins : list, array
        A sequence of altitudes that defines the bins to divide the datasets
        into subplots.
    xlabel, ylabel : str
        String to label the x and y axes.
    s : int, default 10
        Size to render the points^2.
    cmap : str
        A valid matplotlib colormap as a string.
    c : str, tuple
        Color of the scatter points.
    log : bool, default True
        If *True*, the plot will be on log-log axes. If *False*, the plot will
        be on linear axes.
    raster : bool, default True
        Whether to rasterize the points on the plot. Recommended for large 
        numbers of points.
    grid : bool, default True
        If *True*, a gridlines will be plotted.
    suptitle : str
        String to use as the figure's suptitle (like fig.suptitle). This is
        still a bit buggy and may not result in a nice suptitle as desired.
    drawline : bool, default True.
        If *True*, draws a grey 1-to-1 line diagonally across the plot axes.
    columns : int
        Number of columns of subplots. (rows are determined automatically).

    Returns
    -------
    fig : :class:~`matplotlib.figure.Figure`
        The figure instance.
    
    """
    
    fig = plt.figure()
    doFirstPlot = False
    altbins = np.sort(list(set(altbins)))
    
    if not isinstance(c, (str, tuple)):
        raise ValueError('c must be of type str or tuple. \
                          Only a single color can be specified.')
    
    if log:
        x = x.replace(0, np.nan)
        y = y.replace(0, np.nan)
        minvalue = _find_min(x, y)
        limitMagnitude = get_magnitude(minvalue)
        if limitMagnitude == 0:
            lowerLimit = 10**-1
        else:
            lowerLimit = 10**limitMagnitude
    else:
        lowerLimit = 0
        
    maxDataValue = min([max(x.columns), max(y.columns)])
    if maxDataValue != max(altbins):
        altbins = altbins[altbins < maxDataValue] 
        altbins = np.append(altbins, maxDataValue)
    
    minDataValue = min([min(x.columns), min(y.columns)])
    if minDataValue != min(altbins):
        altbins = altbins[altbins > minDataValue]
        doFirstPlot = True
    
    # To get a nice layout for the number of subplots needed, a custom function 
    # is used rather than plt.subplots(rows, columns) (which will plot all 
    # subplots initially) and then removing the ones without data with 
    # " if not len(ax.get_lines()): fig.delaxes(ax) ". This gets complicated 
    # with plotting options (drawline, grid) and adds time in removing the
    # subpots that are not needed.
    subplots = _get_subplot_layout(len(altbins), columns=columns)
    nrows,ncolumns,_ = subplots[0]
        
    if doFirstPlot:
        xfirst = x.loc[:, :altbins[0]]
        yfirst = y.loc[:, :altbins[0]]
        ax = fig.add_subplot(*subplots[0])
        ax = depol_scatter(xfirst, yfirst, s=s, c=c, raster=raster, log=log, ax=ax, 
                            grid=grid, drawline=False)
        ax.set_title('%i to %i m' % (minDataValue, altbins[0]), fontsize='small')
        subplots.pop(0)
        
    for i in xrange(len(altbins)-1):
        xbin = x.loc[:, altbins[i]:altbins[i+1]]
        ybin = y.loc[:, altbins[i]:altbins[i+1]]
        ax = fig.add_subplot(*subplots[i])
        ax = depol_scatter(xbin, ybin, s=s, c=c, raster=raster, log=log, ax=ax, 
                            grid=grid, drawline=False)
        ax.set_title('%i to %i m' % (altbins[i], altbins[i+1]), fontsize='small')
        
    # Handle the last bin of data requested
    if max(altbins) == maxDataValue:
        xlast = x.loc[:, maxDataValue]
        ylast = y.loc[:, maxDataValue]
        ax = depol_scatter(xlast, ylast, s=s, c=c, raster=raster, log=log, ax=ax, 
                            grid=grid, drawline=False)
    
    for ax in fig.get_axes():
        # Get rid of somewhat excessive labels and ticks: 
        # Have ticks and ticklabels only on outer axes and, xlabel/ylabel on 
        # last column/first row only.
        ax.label_outer()
        
        if ax.is_last_row():
            ax.set_xlabel(xlabel)
        if ax.is_first_col():
            ax.set_ylabel(ylabel)
            
        # Reformat the axes lims and drawline manually after plotting:
        ax.set_xlim(lowerLimit, 1)
        ax.set_ylim(lowerLimit, 1)
        if drawline:
            ax.plot([lowerLimit, 1], [lowerLimit, 1], linestyle='-', 
                    lw=1, color='0.5')
    
    # Final adjustments to figure
    fig.set_figwidth(ncolumns*3.5)
    fig.subplots_adjust(hspace=0.2, wspace=0.02)
    if suptitle:
        # Adjust for some extra space for the title
        fig.set_figheight(nrows*3.5 + 0.5*(nrows))
        # Unfortunately, if fig.tight_layout() is used, it will not take into 
        # account the suptitle: 
        #   https://github.com/matplotlib/matplotlib/issues/829
        # One solution would be to use GridSpec instead of the usual subplots.
        # Instead, do a hack (for now) of annotating the first axes with the 
        # suptitle string and set the positioning to 'figure fraction'.
        fig.get_axes()[0].annotate(suptitle, (0.5, 0.98), 
                                     xycoords='figure fraction', ha='center')
    else:
        fig.set_figheight(nrows*3.5)
    
    return fig

def _get_subplot_layout(number, columns=4):
    """Generate the correct number and position of subplots for a figure
    given the total number of subplots needed, and the number of columns. 
    
    author : Annie Seagram
    email : aseagram@eos.ubc.ca
    created : July 09 2013
    
    Parameters
    ----------
    number : int
        Total number of subplots.
    columns : int {optional}
        Number of columns to arrange the subplots.
        Default is 4.
    
    Returns
    -------
    subplots:
        A list of tuples that indicates the position of subplots.
            
    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>> fig1 = plt.figure(1)
    >>> subplots = _get_subplot_layout(4)
    >>> subplots
    [(1, 4, 1), (1, 4, 2), (1, 4, 3), (1, 4, 0)]
    >>> for s in subplots:
            ax = fig1.add_subplot(*s)
    
    """
    
    subplotsIndices = xrange(number)
    rows = int(math.ceil(number / float(columns)))
    subplots = []
    
    if number < columns:
        columns = number
        
    if number % columns == 0:
        for n in xrange(number):
            if n > 0:
                subplots.append((rows, columns, n))
        subplots.append((rows, columns, 0))
    else:
        for n in xrange(number+1):
            if n > 0:
                subplots.append((rows, columns, n))
                
    return subplots

if __name__ == '__main__':

    #------------------------------------
    #  USER OPTIONS and PATHS
    #------------------------------------
    # altitude bin boundaries of how to divide the dataset and make subpots
    altBins = [500, 1000, 2000, 3000, 4500, 6000, 7500, 9000]   
    dpi = 200               # dpi to save the images
    log = True              # True/False for log/linear plots
    raster = True           # rasterize the points in the plots
    labelTitle = True       # For large scatter plot: puts a label of the 
                            # dates as the title
    figwidth = 10           # For the large scatter plot: width/height of 
                            # the figure (square plot)
    
    # Paths to data and files
    root_directory = 'C:\Users\dashamstyr\Dropbox\Python_Scripts\GIT_Repos'
    lnc_path = os.path.join(root_directory, 'CORALNet')
    mpl_path = os.path.join(root_directory, 'SigmaMPL/DATA')
    lnc_filenames = 'LNC_2013-09-09-2013-09-10.h5'
    mpl_filenames = ['201309091353-201309091700_proc.h5', '201309100902-201309102000_proc.h5']
    
    #*******************************************************
    #                       NOTE
    #*******************************************************
    # Under each of the CORALNet and MPL sections 
    # is where you would insert your GUI file selection
    # instead if desired
    #*******************************************************
    
    #----------------------------------------------
    #   Path prep and initialization
    #----------------------------------------------
    output_path = os.path.join(root_directory, 'output')
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        
    _printwidth = 60
    print()
    print('WELCOME'.center(_printwidth))
    print('*'*_printwidth)
    print('Plotting script for MPL vs. CORALNet data'.center(_printwidth))
    print('Plots will be saved in:'.center(_printwidth))
    print(('< %s >' % output_path).center(_printwidth))
    print('*'*_printwidth)
    print()
    
    #---------------------------------------------
    #   CORALNet
    #---------------------------------------------
    print('Getting CORALNet data...')
    
    dtypes = ['BR532','PR532']
    header, df_dict = lnctools.from_HDF(os.path.join(lnc_path, lnc_filenames), dtypes)

    # Get depol data (PR532)
    df = df_dict[dtypes[1]]

    # Get header info
    timestepCN = header['timestep']     # time step of sampling
    minaltCN = header['minalt']         # maximum altitude
    maxaltCN = header['maxalt']         # minimum altitude
    numbinsCN = header['numbins']       # number of altitude bins
    numprofsCN = header['numprofs']     # number of profiles in this file

    # Filter the data if desired here:
    # startdate = dt.datetime(2013, 9, 9, 14)
    # enddate = dt.datetime(2013, 10, 10, 10) 
    # minalt = 150
    # maxalt = 10000
    # df = df.loc[startdate:enddate,:maxalt]
    # df = df.loc[:,:maxalt]
    # if minalt != 0:
    #     df.loc[:,:minalt] = 'nan'

    #---------------------------------------------
    #   MPL
    #---------------------------------------------
    print('Getting MPL data...')
    
    mpl_filepaths = [os.path.join(mpl_path, name) for name in mpl_filenames]

    for f in mpl_filepaths:  
        MPLtemp = mtools.MPL()
        MPLtemp.fromHDF(f)
    
        # This will handle multiple files if needed
        try:
            MPLdata.append(MPLtemp)
        except NameError:
            MPLdata = MPLtemp
    MPLdata.header.sort_index()

    #---------------------------------------------
    #   MPL Resampling
    #---------------------------------------------
    # Resample the MPL data to correspond with the CORALNet data
    altrange = np.linspace(minaltCN, maxaltCN, numbinsCN)

    print()
    print('Resampling altitudes... %i to %i' % (minaltCN, maxaltCN))
    MPLdata = MPLdata.alt_resample(altrange)

    print()
    print('Resampling timesteps... %s' % timestepCN)
    MPLdata = MPLdata.time_resample(timestepCN)

    # Now Get the depolarization data
    depolMPL = MPLdata.depol

    #---------------------------------------------
    #   Filter DEPOL data
    #---------------------------------------------
    print()
    print('Filtering data sets...')
    
    # This is where you would filter out any noise of the depolarization
    # datasets. Here, a simple filter for values 0-1 is used:
    # [ see doc for filter_depol() ]
    
    lnc_filter = filter_depol(df)
    mpl_filter = filter_depol(depolMPL)

    # Note: for two different criteria, be sure to nest the filtered results!
    # lnc_filter = filter_depol(filter_depol(df, f=('le', 1)), f=('gt', 0.01))
    # mpl_filter = filter_depol(filter_depol(depolMPL, f=('le', 1)), f=('gt', 0.01))

    #---------------------------------------------
    #   Dataset information
    #---------------------------------------------
    datetimeLNC = lnc_filter.index
    altLNC = lnc_filter.columns
    altMPL = mpl_filter.columns
    datetimeMPL = mpl_filter.index

    # Find the set of altitudes and dates that are in BOTH of the data sets
    union_alt = altLNC.intersection(altMPL)
    union_datetime = datetimeMPL.intersection(datetimeLNC)

    #---------------------------------------------
    #   Prep for plotting
    #---------------------------------------------
    
    totalPoints = len(union_datetime) * len(union_alt)

    print()
    print('*'*_printwidth)
    print('...Initiating plot sequence...'.center(_printwidth))
    print()
    print('Total number of points to plot:'.center(_printwidth))
    print(('< %i >' % totalPoints).center(_printwidth))
    print()
    print('PLEASE BE PATIENT'.center(_printwidth))
    print()
    print('*'*_printwidth)
    print()
    
    # Get first and last date for labelling if requested
    union_dt_objects = union_datetime.to_datetime()
    firstDate = min(union_dt_objects)
    lastDate = max(union_dt_objects)
    
    # For the file name to save:
    if log:
        logsetting = 'log'
    else:
        logsetting = 'linear'

    #---------------------------------------------
    #   GET DATA
    #---------------------------------------------

    sl = lnc_filter.loc[union_datetime, union_alt]
    sm = mpl_filter.loc[union_datetime, union_alt]
    
    #---------------------------------------------
    #   Scatter plot (large)
    #---------------------------------------------
    print('Preparing scatter plot...')
    
    # Calculate a coefficient of determination (R^2) for a linear regression
    # between the data sets of depolarization values
    sm_flat = np.ravel(sm)
    sl_flat = np.ravel(sl)
    both_mask = np.isfinite(sm_flat) & np.isfinite(sl_flat)
    R = linear_rvalue(sm_flat[both_mask], sl_flat[both_mask])
    Rsquared = R**2

    print('The calculated linear regression value R^2 is:')
    print(Rsquared)

    # Plot ALL of the data to one figure
    # Do one with colours for altitudes, and one with just black points

    altitudes = list(np.tile(union_alt, len(union_datetime)))

    fig3 = plt.figure(3, figsize=(figwidth, figwidth))
    ax3 = depol_scatter(sm, sl, log=log, c=altitudes, altbins=altBins, altunit='km')
    
    fig4 = plt.figure(4, figsize=(figwidth, figwidth))
    ax4 = depol_scatter(sm, sl, log=log, c='k', altbins=altBins, altunit='km')
    
    for ax in [ax3, ax4]:
        ax.annotate('$R^2$ = %.g' % Rsquared, (0.8, 0.1), 
                    xycoords='axes fraction', bbox=dict(boxstyle="round", 
                    fc="0.9")
                    )
        ax.set_aspect(1)
        ax.set_xlabel('MPL')
        ax.set_ylabel('CORALNet')

        if labelTitle:
            ax.set_title('{0} to {1}'.format(firstDate, lastDate))

    fig3.savefig(os.path.join(output_path, 'depol_full_byalt_%s.png' % logsetting), dpi=dpi)
    fig4.savefig(os.path.join(output_path, 'depol_full_%s.png' % logsetting), dpi=dpi)
    
    print('... FINISHED two scatter plots!')
    print()
    
    #---------------------------------------------
    #   Scatter plot (by altitudes)
    #---------------------------------------------
    print('Preparing scatter plot by altitudes...')
    print('Altitude bins requested:')
    print(altBins)
    
    figtitle = '{0} to {1}'.format(firstDate, lastDate)
    fig1 = depol_scatter_byalt(sm, sl, altBins, 'MPL', 'CORALNet', 
                            suptitle=figtitle, log=log, columns=4)
    
    # tight_layout() is especially handy for this figure given certain 
    # row/column configurations
    fig1.tight_layout()
    fig1.savefig(os.path.join(output_path, 'subplots_byalt_%s.png' % logsetting), dpi=dpi)
    
    print('... FINISHED scatter by altitude!')
    print()
    
    #---------------------------------------------
    #   Difference (space/time)
    #---------------------------------------------
    print('Preparing difference plot...')
    print()
    print('Rendering a diff. plot with (CORALNet - MPL)...')
    print('i.e +ve values mean CORALNet values are < HIGHER > than MPL')
    print('    -ve values mean CORALNet values are < LOWER >  than MPL')
    print()

    fig2 = plt.figure(2, figsize=(10, 5))
    ax2 = fig2.add_subplot(111)
    depol_diff(sm.T, sl.T, union_datetime, union_alt, vlim=0.04, ax=ax2)

    fig2.savefig(os.path.join(output_path, 'diff.png'), dpi=dpi)
    
    print('... FINISHED diff plot!')
    print()
    
    #---------------------------------------------
    #   Exit
    #---------------------------------------------
    print('*'*_printwidth)
    print('Finished all plots'.center(_printwidth))
    print('*'*_printwidth)
    print('GOODBYE'.center(_printwidth))