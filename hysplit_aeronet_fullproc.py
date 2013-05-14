import hysplit_tools as tools
import hysplit_traceback_single as traceback
import file_process_tools as proc
import numpy as np
import scipy.io
import os,sys

#STEP #1: Filter original Hysplit files to include only days that have aeronet data

level = '15'

aerofilt_dir = 'K:\Hysplit\AEROFILT'
os.chdir(aerofilt_dir)

startdir = os.getcwd()

aeronet_file = tools.get_files('Choose Aeronet File',('Aeronet Files','*.Dubovik'))

hysplit_path = tools.set_dir('Select Hysplit Directory')

hydir = os.path.split(hysplit_path)[1]

tools.aeronet_dayfilter(level,aeronet_path[1:-1],aerofilt_dir,hysplit_path)

#STEP #2: Convert aeronet file and filtered hysplit files to .mat format

tools.hysplit_matfile_generator(level,hydir,aerofilt_dir)

tools.aeronet_matfile_generator(level,aeronet_file[1:-1],hydir,aerofilt_dir)

#STEP #3: Generate Hysplit traceback file
 
traceback.traceback(level, hydir, aerofilt_dir)

#Step #4: Process and combine Aeronet and Hysplit files

proc.aeroproc(level,aerofilt_dir)

proc.traceproc(level,aerofilt_dir)

proc.combproc(level,aerofilt_dir)
