import hysplit_tools_v2 as tools
import hysplit_traceback_single as traceback
import file_process_tools as proc
import numpy as np
import scipy.io
import os,sys


#STEP #1: Filter original Hysplit files to include only days that have aeronet data

aerofilt_topdir = 'F:\Hysplit\AEROFILT'
os.chdir(aerofilt_topdir)

startdir = os.getcwd()

aeronet_path = tools.get_files('Choose Aeronet File',('Aeronet Files','*.dubovik'))

hysplit_path = tools.set_dir('Select Hysplit Directory')

aerofilt_dir = tools.aeronet_dayfilter(aeronet_path[1:-1],aerofilt_topdir,hysplit_path)

#STEP #2: Convert aeronet file and filtered hysplit files to .mat format

tools.hysplit_matfile_generator(aerofilt_dir)

tools.aeronet_matfile_generator(aeronet_path[1:-1],aerofilt_dir)

#STEP #3: Generate Hysplit traceback file
 
traceback.traceback(aerofilt_dir)


