# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 15:43:23 2013

@author: dashamstyr
"""

import numpy

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 

    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window must be either 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=numpy.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=numpy.ones(window_len,'d')
    else:
        w=eval('numpy.'+window+'(window_len)')

    y=numpy.convolve(w/w.sum(),s,mode='valid')
    return y[(window_len/2-1):-(window_len/2)]

def gauss_kern(sizex, sizey=None):
    """ Returns a normalized 2D gauss kernel array for convolutions """
    import scipy as sci 
    sizex = int(sizex)
    if not sizey:
        sizey = sizex
    else:
        sizey = int(sizey)
    x, y = sci.mgrid[-sizex:sizex+1, -sizey:sizey+1]
    g = sci.exp(-(x**2/float(sizex)+y**2/float(sizey)))
    return g / g.sum()


def blur_image(im, (nx,ny), kernel = 'Gaussian'):
    """ blurs the image by convolving with a gaussian kernel of typical
        size n. The optional keyword argument ny allows for a different
        size in the y direction.
    """
    from scipy import signal
    import numpy as np
    
    if kernel == 'Gaussian':        
        g = gauss_kern(nx, sizey=ny)
    elif kernel == 'Flat':
        g = np.ones((nx,ny))
        g = g / g.sum()
    else:
        raise ValueError, "Kernel must be either Gaussian or Flat"
        
    improc = signal.convolve(im,g, mode='same')
    return improc

