import pandas as pan
import numpy as np
import os,sys
import LNC_tools_v2 as LNC
import LNC_plot_v1 as LNCplot

#----------------------------------------------------------------------------
#Uses tools created in LNC_tools to open all files in a folder and resample
#them to a regular spacing in altitude/date the concatenates them into one
#pandas dataframe and plots it using LNC_plot
#July 05, 2010
#----------------------------------------------------------------------------

olddir = os.getcwd()

#os.chdir('C:\Users\User\Documents\CORALNet\Data\ASCII Files')

newdir = LNC.set_dir('Select Event Folder')

os.chdir(newdir)

files = os.listdir(newdir)
maskfiles = []
datafiles = []
procfiles = []
rawfiles = []

#set altitude range and date step sizes

altrange = np.arange(10,15010,10)#meters
timestep = '120S' #seconds

#check to see if each file has been processed before and separate processed
#files into a new list

for f in files:
    if '_proc' in f or '.pickle' in f:
        procfiles.append(f)
    else:
        rawfiles.append(f)

#search through list of files to separate fiels to be used as a mask from those
#with data to be plotted
#initially, mask files are designated BR1064 for 1064nm Backscatter Ratio

for f in rawfiles:
    if 'BR1064' in f:
        maskfiles.append(f)
    else:
        datafiles.append(f)

#make sure the files are in a common order of ascending date (assuming they're all
#from the same station
maskfiles.sort()
datafiles.sort()

#first check to make sure the same number of files in each list

if len(maskfiles) != len(datafiles):
    sys.exit("Error: Mask files don't match data files")

#double check to make sure the mask files match up with the data files
for d,m in zip(datafiles, maskfiles):
    [d_stat,d_date,d_type] = d.split('_')
    [m_stat,m_date,m_type] = m.split('_')
    print 'Checking mask/data match for %s'%(d_date)
    if d_date == m_date and d_stat == m_stat:
        print 'Check!'
        continue
    else:
        sys.exit("Error: Mask files don't match data files")

#open, altitude resample, and concatenate data and mask files

for d,m in zip(datafiles, maskfiles):
    d_temp, data_prod = LNC.lnc_reader(d)
    d_realt = LNC.alt_resample(d_temp,altrange)

    try:
        d_event = pan.concat([d_event,d_realt])
    except NameError:
        d_event = d_realt

    m_temp, data_prod = LNC.lnc_reader(m)
    m_realt = LNC.alt_resample(m_temp,altrange)

    try:
        m_event = pan.concat([m_event,m_realt])
    except NameError:
        m_event = m_realt

    
#sort by index to make certain data is in order then set date ranges to match

d_event = d_event.sort_index()
m_event = m_event.sort_index()

start = d_event.index[0]
end = d_event.index[-1]

d_event = LNC.time_resample(d_event,timestep,[start,end])
m_event = LNC.time_resample(m_event,timestep,[start,end])


dfmask = LNC.BR_mask(m_event,d_event)

d_filename = datafiles[0].split('.')[0]+'-'+datafiles[-1].split('.')[0]
d_event.save(d_filename+'.pickle')

m_filename = maskfiles[0].split('.')[0]+'-'+maskfiles[-1].split('.')[0]
m_event.save(m_filename+'.pickle')

dfmask.save(d_filename+'_masked.pickle')




