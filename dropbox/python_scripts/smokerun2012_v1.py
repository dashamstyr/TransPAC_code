import sys,os
import hysplit_tools_v2 as tools
import subprocess

# create list of stations in question
UBC = ('UBC',49.256,-123.250) #done
Whistler = ('WHI',50.128,-122.95) #done


stations = [UBC]#,Whistler]

#set run time
run_hours = '-150'

#set heights
heights = range(1600,4050,50)

#set dates and times
year = '12'
month = '07'
day = [7]#range(5,10)
hour = range(0,24,6)

totalruns = len(stations)*len(day)*len(hour)*len(heights)
runs = 0

#select meteorology files

olddir = os.getcwd()
os.chdir('C:\hysplit4\meteo')
meteo_list = tools.get_files('Select Meteorology Files')
meteo_files = meteo_list.split()

#set output directory

os.chdir('C:\Hysplit')
output_dir = tools.set_dir('Select Output Directory')

for s in stations:
    for d in day:
        for h in hour:
            for z in heights:
                start_time = [year,month,d,h]
                    
                tools.control_single(s,start_time,run_hours,z,meteo_files,output_dir)
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                
                proc = subprocess.call('c:/hysplit4/exec/hyts_std', startupinfo=startupinfo)
                runs += 1
                complete = 100.0*runs/totalruns
                print_time = ' '
                for n in start_time:
                    print_time += str(n)
                print s[0]+' '+print_time+' '+str(z)+'m'
                print str(complete)+'% complete'
                    

os.chdir(olddir)
            
