def mplreader(filename):

    import numpy as np
    import array
    import datetime as dt

    binfile = open(filename,'rb')

    #create a class instance to contain the data and header

    class MPL:
        pass

    
    d = MPL()
    
    integers = array.array('I')
    floats = array.array('f')

    #start by getting first 16 header inputs
    integers.fromfile(binfile, 16)

    d.header.unitnum = integers[0]
    d.header.version = integers[1]
    year = integers[2]
    month = integers[3]
    day = integers[4]
    hour = integers[5]
    minute = integers[6]
    second = integers[7]

    d.header.datetime = dt.datetime(year,month,day,hour,minute,second)

    d.header.shotsum = integers[8]  #total number of shots collected
    d.header.trigfreq = integers[9] #laser trigger frequency (usually 2500 Hz)
    d.header.energy = integers[10]  #mean of laser energy monitor in mW(?)
    d.header.temp_0 = integers[11]  #mean of A/D#0 readings*100
    d.header.temp_1 = integers[12]  #mean of A/D#1 readings*100
    d.header.temp_2 = integers[13]  #mean of A/D#2 readings*100
    d.header.temp_3 = integers[14]  #mean of A/D#3 readings*100
    d.header.temp_4 = integers[15]  #mean of A/D#4 readings*100

    #next two inputs are floats
    floats.fromfile(binfile, 2)

    d.header.bg_avg1 = floats[0] #mean background signal value for channel 1
    d.header.bg_std1 = floats[1] #standard deviation of backgruond signal for channel 1

    #then two integers
    integers.fromfile(binfile, 2)

    d.header.numchan = integers[16] #number of channels
    d.header.numbins = integers[17] #total number of bins per channel

    #two more floats
    floats.fromfile(binfile, 2)

    d.header.bintime = floats[2]  #bin width in seconds
    d.header.rangecal = floats[3] #range offset in meters, default is 0
    
    #three integers
    integers.fromfile(binfile, 3)

    d.header.databins = integers[18]  #number of bins not including those used for background
    d.header.scanflag = integers[19]  #0: no scanner, 1: scanner
    d.header.backbins = integers[20]  #number of background bins

    floats.fromfile(binfile, 8)

    d.header.az = floats[4]  #scanner azimuth angle
    d.header.el = floats[5]  #scanner elevation angle
    d.header.deg = floats[6] #compass degrees (currently unused)
    d.header.pvolt0 = floats[7] #currently unused
    d.header.pvolt1 = floats[8] #currently unused
    d.header.gpslat = floats[9] #GPS latitude in decimal degreees (-999.0 if no GPS)
    d.header.gpslon = floats[10]#GPS longitude in decimal degrees (-999.0 if no GPS)
    d.header.cloudbase = floats[11] #cloud base height in [m]

    integers.fromfile(binfile, 2)

    d.header.baddat = integers[21]  #0: good data, 1: bad data
    d.header.version = integers[22] #version of file format.  current version is 1

    floats.fromfile(binfile, 2)

    d.header.bg_avg2 = floats[12] #mean background signal for channel 2
    d.header.bg_std2 = floats[13] #mean background standard deviation for channel 2

    integers.fromfile(binfile, 6)

    d.header.mcs = integers[23]  #MCS mode register  Bit#7: 0-normal, 1-polarization
                                 #Bit#6-5: polarization toggling: 00-linear polarizer control
                                 #01-toggling pol control, 10-toggling pol control 11-circular pol control

    d.header.firstbin = integers[24]  #bin # of first return data
    d.header.systype = integers[25]   #0: standard MPL 1: mini MPL
    d.header.syncrate = integers[26]  #mini-MPL only, sync pulses seen per second
    d.header.firstback = integers[27] #used for mini-MPL first bin used for background calcs
    d.header.headersize2 = integers[28] #currently unused

    #now use number of bins from above to split data into channels

    floats.fromfile(binfile, -1)

    d.data.channel = range(d.header.numchan)

    for b in d.data.channel:
        for n in range(d.header.numbins):
            d.data.channel[b].append[n]
    
    
    return datain


if __name__=='__main__':

    import os, sys

    olddir = os.getcwd()

    os.chdir('C:\Users\dashamstyr\Dropbox\China!\MPL Data')

    filename = '201301031600.mpl'

    data = mplreader(filename)

    os.chdir(olddir)
