# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 13:02:55 2014

@author: dashamstyr
"""

import os,sys

toplevel = 

historyfile = "MPL_history_file.txt"
locationsfile = "MPL_locations.html"
mainpage = "index.html"

with open(historyfile, 'r') as hist:
    lines = hist.readlines()
    
    locations = []
    start_dates = []
    end_dates = []
    
    for l in lines:
        if "Location" in l:
            locations.append(l.split(": ")[1])
        if "Start_date" in l:
            start_dates.append(l.split(": ")[1])
        if "End_date" in l:
            try:
                end_dates.append(l.split(": ")[1])
            except ValueError:
                end_dates.append("na")
    
    headerdict = {}    
    for loc,start,end in locations,start_dates,end_dates:
        key = loc
        val = (start,end)
        headerdict[key] = val
 
os.chdir       

    
oldline = '<h1> Image Archive </h1>'

for loc, dates in headerdict.iteritems():
    newline = '<a href="'+loc+'-bymonth.html">'+loc+'</a>'
    with open(mainpage, 'w') as htmlfile:
        mainlines = htmlfile.readlines()
        
        for l in mainlines:
            if oldline in l:
                
        
        
    