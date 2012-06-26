def improve_reader(filepath):
    #---------------------------------------------------------------------------
    #This program opens an ASCII file downloaded from IMPROVE and outputs a
    #Pandas timeseries dataframe with the data and relevant metadata
    #---------------------------------------------------------------------------
    import pandas as pan
    import numpy as np
    
    print 'Processing '+filepath

    df = pan.read_csv(filepath, index_col=[1,2], parse_dates = True)

    del df['POC']
    del df['Dataset']

    return df

    
if __name__=='__main__':
    import os
    import pandas as pan
    os.chdir('C:\Users\user\Dropbox\TransPAC2010\IMPROVE Data')

    files = os.listdir(os.getcwd())

    for f in files:
        [fname,fext] = f.split('.')
        if fext == 'txt':
            try:
                df = pan.concat([df,improve_reader(f))
            except NameError:
                df = improve_reader(f)
        
    df.save('IMPROVE_data_all.pickle')
            
