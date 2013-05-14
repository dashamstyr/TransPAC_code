# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 11:28:50 2013

@author: User
"""

def lhc_1D(dist, distparams, n = 100):
    
    import numpy as np
    from scipy import stats
    #make sure distribution parameters are floats
    distparams = tuple([float(k) for k in distparams])
    
    step = 1.0/n
    
    #pick n evenly distributed divisions
    divisions = np.arange(0.,1.,step)
    
    #randomize the points
    np.random.shuffle(divisions)
    
    sample = np.empty(n, dtype='float')
    lhc_out = np.empty_like(sample)
    
    #use uniform distribution to select 
    for i,d in enumerate(divisions):
        sample[i] = stats.uniform(d,step).rvs()
        lhc_out[i] = dist(*distparams).ppf(sample)
    
    return lhc_out

def lhc_sample(samp_in, size = 100):
    
    import numpy as np
    from scipy import stats
    
    if isinstance(size, (list,tuple)):
        n = np.product(size)
    else:
        n = size
        
    step = 100.0/n
    
    percentage = np.arange(0.0,100.0,step)
    
    np.random.shuffle(percentage)
    
    lhc_out = np.empty(n)
    
    for i,p in enumerate(percentage):
        samp_rnd = stats.uniform(p,step).rvs()
        lhc_out[i] = stats.scoreatpercentile(samp_in,samp_rnd)
    
    if isinstance(size, (list,tuple)):
        lhc_out.shape = size
    
    return lhc_out

if __name__ == '__main':
    from scipy import stats
    import numpy as np
    import matplotlib.pyplot as plt
    
    dist = stats.norm
    
    params = (50,1)
    
    c1 = lhc_1D(dist,params,100)
    c2 = lhc_1D(dist,params,100)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(c1,10)
    plt.show()
    