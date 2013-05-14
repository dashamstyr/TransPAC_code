
def control_single(location,start_time,run_hours,height,meteo_files,output_dir):
    import os,sys
    
    os.chdir('c:/hysplit4/working/')
    if os.path.isfile('CONTROL'):
        os.remove('CONTROL')
    year = str(start_time[0]).zfill(2)
    month = str(start_time[1]).zfill(2)
    day = str(start_time[2]).zfill(2)
    hour = str(start_time[3]).zfill(2)

    time_in = year+' '+month+' '+day+' '+hour
    latlonht = str(location[1])+' '+str(location[2])+' '+str(height)         
    output_path = output_dir+'/'+location[0]+'/'

    if not os.path.isdir(output_path):
        os.makedirs(output_path)
                       
    
    output_file = location[0]+year+month+day+hour+'_'+str(height)+'.txt'
    cont_file = open('CONTROL','w')
    cont_file.writelines(time_in+'\n'+'1\n'+latlonht+'\n'+str(run_hours)+'\n'+\
                         '0 \n10000\n'+str(len(meteo_files))+'\n')

    for n in meteo_files:
        [meteo_path,meteo_filename] = os.path.split(n)
        cont_file.writelines(meteo_path+'/\n'+meteo_filename+'\n')

    cont_file.writelines(output_path+'\n'+output_file)
    cont_file.close()

##def infile_generator(folder,basename):
##    import os,sys
##
##    os.chdir(folder)
##    if os.path.isfile('INFILE'):
##        os.remove('INFILE')
##
##    c = os.listdir(os.getcwd())
##
##    for f in c:
        


def set_dir(titlestring):
    from Tkinter import *
    import tkFileDialog
     
     
    master = Tk()
    master.withdraw() #hiding tkinter window
     
    file_path = tkFileDialog.askdirectory(title=titlestring)
     
    if file_path != "":
       return str(file_path)
     
    else:
       print "you didn't open anything!"
     
def get_files(titlestring,filetype = ('.txt','*.txt')):
    from Tkinter import *
    import tkFileDialog
     
     
    master = Tk()
    master.withdraw() #hiding tkinter window

    file_path = []
     
    file_path = tkFileDialog.askopenfilename(title=titlestring, filetypes=[filetype,("All files",".*")],multiple='True')
     
    if file_path != "":
       return str(file_path)
     
    else:
       print "you didn't open anything!"
       
def aeronet_import(filename):
    # import aeronet data file into a list of dictionaries
    # filename must include full path to file
    
    import csv

    filetoread = open(filename,'r')
    header = []
    headerlines = 3
    for n in range(0,headerlines):
        header.append(next(filetoread))

    tempdict = csv.DictReader(filetoread)

    aerodict = []
    for row in tempdict:
        aerodict.append(row)

    filetoread.close()

    return header,aerodict

def hysplit_import(filename):
    # import aeronet data file into a list of dictionaries
    # filename must include full path to file
    
    import csv
    import numpy as np

    filetoread = open(filename,'rb')
    header = []
    headerlines = 11
    for n in range(0,headerlines):
        header.append(next(filetoread))

    temp = csv.reader(filetoread,delimiter = ' ',skipinitialspace = 'True')

    hysplitdata = np.array([])
    for row in temp:
        hysplitdata = np.append(hysplitdata,row)
    hysplitdata = np.reshape(hysplitdata,(-1,13))
    hysplitdata = hysplitdata.astype('float16')

    filetoread.close()  
    
    return header,hysplitdata

def aeronet_extract(aerofile,filterkeys):
    #import data dictionary and filter for desired keys
    #output goes into matlab file (filename.mat) in selected output folder
    #with each key pointing to a list of values date time and data type (1.5 or 2.0) are included
    #filterkeys is a list of variables to extract from the Aeronet file
    
    import os,sys
    
    [header,aerodict] = aeronet_import(aerofile)
    
    datelist = []
    datatypelist = []
    output_dict = dict()
    for line in aerodict:
        tempdate = line['Date(dd-mm-yyyy)'].split(':')
        temptime = line['Time(hh:mm:ss)'].split(':')
        year = int(tempdate[2][-2:])
        month = int(tempdate[1])
        day = int(tempdate[0])
        hour = float(temptime[0])
        minute = float(temptime[1])/60
        second = float(temptime[2])/3600
        rounded_hour = int(round(hour+minute+second))
        tempdatatype = line['DATA_TYPE']

        date = [year,month,day,rounded_hour]
        datelist.append(date)
        datatypelist.append(tempdatatype)

    output_dict['Date'] = datelist
    output_dict['Data Type'] = datatypelist

    for key in filterkeys:
        temp = []
        for line in aerodict:
            temp.append(float(line[key]))
            

        output_dict[key] = temp

    return output_dict
                    
def aeronet_dayfilter(level,aeronet_path,aerofilt_dir,hysplit_path):
    import os,sys
    import shutil

    startdir = os.getcwd()
    
    [aeronet_dir,aeronet_file] = os.path.split(aeronet_path)

    os.chdir(aeronet_dir)

    [aeronet_header,aerodict] = aeronet_import(aeronet_file)

    [hysplit_dir,hysplit_folder] = os.path.split(hysplit_path)
    
    aerodir = aerofilt_dir+'/Level '+level+'/'+hysplit_folder
    
    try:
        os.mkdir(aerodir)
    except OSError:
        pass
    
    
    datestring = []
    for line in aerodict:
        tempdate = line['Date(dd-mm-yyyy)'].split(':')
        year = tempdate[2][-2:]
        month = tempdate[1]
        day = tempdate[0]
        datestring.append(year+month+day)
    
    os.chdir(hysplit_path)
    hysplit_files = os.listdir('.')
    for f in hysplit_files:
        for s in datestring:
            if s in f:
                shutil.copy2(f,aerodir)

    os.chdir(startdir)

    return aerodir

def haversine(lat1,long1,lat2,long2):
    #function to compute distance and bearing between
    #two latitude/longitude coordinates
    #output distance is in km, bearing is in degrees
    
    from math import *
    
    R = 6371 #radius of Earth in km
    dlat = radians(lat2-lat1)
    dlong = radians(long2-long1)
    lat1 = radians(lat1)
    lat2 = radians(lat2)

    a = sin(dlat/2)**2 + sin(dlong/2)**2 * cos(lat1) * cos(lat2)
    b = 2*atan2(sqrt(a),sqrt(1-a))

    d = R*b

    y = sin(dlong)*cos(lat2)
    x = cos(lat1)*sin(lat2) - sin(lat1)*cos(lat2)*cos(dlong)

    theta = degrees(atan2(y,x))

    return d,theta

def ellipserad(a,b,theta1,theta2):
    #function to output distance from center to edge of a tilted ellipse
    #a = major axis
    #b = minor axis
    #theta1 = heading between center and point in Earth-centerd coordinates (0 deg = N)
    #theta2 = angle between major axis and N-S axis
    
    
    from math import *

    dtheta = theta2-theta1

    x = b*cos(radians(dtheta))
    y = a*sin(radians(dtheta))

    r = a*b/sqrt(x**2 + y**2)

    return r

def matfile_test(filename,filelist):

    nametest1 = filename.split('.')

    if nametest1[-1] == 'txt':

        for htest in filelist:

            nametest2 = htest.split('.')

            if  nametest2[-1] == 'mat':
                return False
        return True
    else:
        return False

def hysplit_matfile_generator(level,station,aerofilt_dir):
    #tool for extracting data from hysplit trajectory files and putting it into
    #mat files for storage in float16 format
    import numpy as np
    import scipy.io
    import os,sys

    data_cats = ('year','month','day','hour','delta_t','lat','lon','alt','press')


    startdir = os.getcwd()

    hysplit_folder = aerofilt_dir+'\Level '+level+'\\'+station
    os.chdir(hysplit_folder)

    hysplit_files = os.listdir(os.getcwd())
                    
    for h in hysplit_files:

        if matfile_test(h,hysplit_files):

            #import hysplit text file
            [head,data] = hysplit_import(h)

            #create dictionary with {varname: array} based on column names

            output_data = [data[:,2],data[:,3],data[:,4],data[:,5],data[:,8],\
                           data[:,9],data[:,10],data[:,11],data[:,12]]

            output_dict = dict(zip(data_cats,output_data))

            #use scipy.savemat to save it as a .mat file

            scipy.io.savemat(h,output_dict)

    os.chdir(startdir)        

def aeronet_matfile_generator(level,aerofile,station,aerofilt_dir):
    import os, sys
    import scipy.io
    import numpy as np

    startdir = os.getcwd()

    output_path = aerofilt_dir+'\Level '+level+'\\'+station

    output_folder = station

    numdist_keys = ['0.050000','0.065604','0.086077','0.112939','0.148184','0.194429','0.255105',\
               '0.334716','0.439173','0.576227','0.756052','0.991996','1.301571','1.707757',\
               '2.240702','2.939966','3.857452','5.061260','6.640745','8.713145','11.432287','15.000000']

    keylist = ['Inflection_Point[um]','VolCon-T','EffRad-T','VolMedianRad-T','StdDev-T',\
                        'VolCon-F','EffRad-F','VolMedianRad-F','StdDev-F',\
                        'VolCon-C','EffRad-C','VolMedianRad-C','StdDev-C']

    filename = 'Aerostats_'+output_folder+'_'+level

    newdict = aeronet_extract(aerofile,keylist)

    temp_dist = []
    numdist_diameters = []
    for tempkey in numdist_keys:
        tempdict = aeronet_extract(aerofile,[tempkey])
        temp_dist.append(tempdict[tempkey])
        numdist_diameters.append(float(tempkey))


    newdict['Numdist'] = np.array(temp_dist, dtype='float16')
    newdict['Diameters'] = numdist_diameters

    os.chdir(output_path)
            
    scipy.io.savemat(filename,newdict)

    os.chdir(startdir)
