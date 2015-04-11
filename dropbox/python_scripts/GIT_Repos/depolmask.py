# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 12:21:26 2013

@author: User
"""
import os, sys

import pandas as pan
import datetime
import numpy as np


lib = 'C:\\Users\\dashamstyr\\Dropbox\\Python_Scripts'
datalib = 'C:\\Users\\dashamstyr\\Dropbox\\Lidar Files\\UBC Cross-Cal\\Processed'
figurelib = 'C:\\Users\\dashamstyr\\Dropbox\\Lidar Files\\UBC Cross-Cal\\Figures'

dataname = 'scatterdat1128.h5'
try:
    sys.path.append(os.path.join(lib, 'LNCcode'))
    sys.path.append(os.path.join(lib, 'MPLcode'))
    from LNCcode import LNC_tools as ltools
    from MPLcode import MPLtools as mtools
    import coDepol as cod
except ImportError:
    raise Exception('You havent specified where your modules are located!')

import matplotlib.pyplot as plt


os.chdir(datalib)

lncfile = ltools.get_files('Select Masked LNC File',filetype = ('.h5','*.h5'))

lnc_header,lnc_data= ltools.from_HDF(lncfile[0],['PR532_msk'])

lnc_masked = lnc_data['PR532_msk']

mplfile = mtools.get_files('Select MPL file', filetype = ('.h5', '*.h5'))

mpl_data = mtools.MPL()

mpl_data.fromHDF(mplfile[0])



minalt = 150
maxalt = 15000
altstep = 30

altrange = np.arange(minalt,maxalt,altstep)

#mpl_data.alt_resample(altrange)
#lnc_masked = ltools.alt_resample(lnc_masked,altrange)

try:
    mpl_drat = mpl_data.depolrat[0]
except IndexError:
    mpl_data.calculate_depolrat()
    mpl_drat = mpl_data.depolrat[0]



datetimeMPL = mpl_drat.index
datetimeLNC = lnc_masked.index


union_datetime = datetimeMPL.intersection(datetimeLNC)

mpl_event = mpl_drat.loc[union_datetime]
lnc_event = lnc_masked.loc[union_datetime]

def maskmaker(x):
    if np.isnan(x):
        return x
    else: x = 1
    return x

mplmask = mpl_event.applymap(maskmaker)
lncmask = lnc_event.applymap(maskmaker)

mpl_event = mpl_event*lncmask
lnc_event = lnc_event*mplmask

store = pan.HDFStore(dataname)
    
store['MPL'] = mpl_event
store['LNC'] = lnc_event
    
store.close()

mpl_flat = np.ravel(mpl_event)
lnc_flat = np.ravel(lnc_event)



both_mask = np.isfinite(mpl_flat) & np.isfinite(lnc_flat)
R = cod.linear_rvalue(mpl_flat[both_mask], lnc_flat[both_mask])
Rsquared = R**2

m,b = np.polyfit(lnc_flat[both_mask],mpl_flat[both_mask],1)

print m,b

linfit = np.polyval([m,b],lnc_flat[both_mask])



print('The calculated linear regression value R^2 is:')
print(Rsquared)

# Plot ALL of the data to one figure

#altitudes = list(np.tile(union_alt, len(union_datetime)))
os.chdir(figurelib)
fig2 = plt.figure(2, figsize=(10, 5))
ax2 = fig2.add_subplot(111)
cod.depol_diff(lnc_event.T, mpl_event.T, union_datetime, mpl_event.columns, vlim=0.5, ax=ax2)

fig2.savefig('testdiff.png')

print "Making Figure"
fig3 = plt.figure(3, figsize=(10,10))
ax3 = cod.depol_scatter(lnc_flat.T, mpl_flat.T, log=False)
#ax3.plot(lnc_flat[both_mask],linfit,'r--')

#ax3.annotate('$R^2$ = %.g' % Rsquared, (0.8, 0.1), 
#            xycoords='axes fraction', bbox=dict(boxstyle="round", 
#            fc="0.9")
#            )
ax3.set_aspect(1)
ax3.set_xlabel('CORALNet')
ax3.set_ylabel('MPL')

altBins = [150,6000, 13000, 15000]  

altitudes = list(np.tile(mpl_event.columns, len(union_datetime)))

fig4 = plt.figure(4, figsize=(10,10))
ax4 = cod.depol_scatter(lnc_flat, mpl_flat, log=False, c=altitudes, altbins=altBins, altunit='km')
#ax4.annotate('$R^2$ = %.g' % Rsquared, (0.8, 0.1), 
#            xycoords='axes fraction', bbox=dict(boxstyle="round", 
#            fc="0.9")
#            )
ax4.set_aspect(1)
ax4.set_xlabel('CORALNet')
ax4.set_ylabel('MPL')

print "Displaying Figure"
plt.draw()
print "Done"
#if labelTitle:
#    ax.set_title('{0} to {1}'.format(firstDate, lastDate))
#
fig2.savefig('testdiff.png')
fig3.savefig('testscat.png')
fig4.savefig('testscat_alt.png')