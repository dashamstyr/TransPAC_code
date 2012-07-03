def improve_reader(filepath):
    #---------------------------------------------------------------------------
    #This program opens an ASCII file downloaded from IMPROVE and outputs a
    #Pandas timeseries dataframe with the data and relevant metadata
    #---------------------------------------------------------------------------
    import pandas as pan
    import numpy as np
    
    print 'Processing '+filepath

    df = pan.read_csv(filepath, index_col=[1,2], parse_dates = True)

    #delete these two columns that aren't used iun future operations
    del df['POC']
    del df['Dataset']

    #rename columns with simpler names that work better in graphing
    oldcols = ['ALf:Value','CAf:Value','FEf:Value','MF:Value','Kf:Value',
               'SIf:Value','TIf:Value']
    newcols = ['Al','Ca','Fe','PM 2.5','K','Si','Ti']
    
    colnames = dict(zip(oldcols,newcols))
    df.rename(columns=colnames, inplace=True)

    return df

def dayfilter(df, start, end):
    #function that filters an IMPROVE dataframe by date and returns a new
    #dataframe containing ony data collected within those dates
    #assumes dataframe has heirarchical index ith dates on level 1
    #and start and end must be datetime objects
    import numpy as np

    for row_name, row in df.iterrows():
        timetag = row_name[1]
        colnames = row.index
        if start <= timetag <= end:
            try:
                tempdat = np.vstack([tempdat, row.values])
                index.append(row_name)
            except NameError:
                tempdat = row.values
                index = [row_name]

    dfout = pan.DataFrame(data = tempdat, index = index, columns = colnames)

    return dfout

    
if __name__=='__main__':
    import os
    import pandas as pan
    from datetime import datetime
    os.chdir('C:\Users\dashamstyr\Dropbox\TransPAC2010\IMPROVE Data')

    files = os.listdir(os.getcwd())

    for f in files:
        [fname,fext] = f.split('.')
        if fext == 'txt':
            try:
                df = pan.concat([df,improve_reader(f)])
            except NameError:
                df = improve_reader(f)

    stat_dat = {'HACR1':(datetime(2010,03,24),datetime(2010,04,29)),
                'DENA1':(datetime(2010,03,21),datetime(2010,04,29)),
                'REDW1':(datetime(2010,03,24),datetime(2010,04,29)),
                'SNPA1':(datetime(2010,03,24),datetime(2010,04,29)),
                'GLAC1':(datetime(2010,03,24),datetime(2010,04,29)),
                'BADL1':(datetime(2010,03,27),datetime(2010,04,29)),
                'VOYA2':(datetime(2010,03,31),datetime(2010,04,29)),
                'EGBE1':(datetime(2010,04,02),datetime(2010,04,29)),
                'LYBR1':(datetime(2010,04,02),datetime(2010,05,02)),
                'ACAD1':(datetime(2010,04,05),datetime(2010,05,05))}
    
    grouped = df.groupby(level=0)

    for group in grouped:
        timerange = stat_dat[group[0]]
        temp = dayfilter(group[1],timerange[0],timerange[1])
        
        try:
            df_filt = pan.concat([df_filt,temp])
        except NameError:
            df_filt = temp
        
    df.save('IMPROVE_data_all.pickle')
    df_filt.save('IMPROVE_data_April.pickle')
            
