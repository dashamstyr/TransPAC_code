def Improve_plot(filepath):
    import os
    import pandas as pan
    from matplotlib import pyplot as plt

    os.chdir(filepath)
    df = pan.load('IMPROVE_data_all.pickle')

    return df
    #fig = plt.figure()

    






if __name__ == '__main__':

    f = 'C:\Users\user\Dropbox\TransPAC2010\IMPROVE Data'

    df = Improve_plot(f)
