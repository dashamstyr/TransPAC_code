def depol_plot(xdata, ydata, data):
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors

    xaxis, xticks = dateticks(xdata)
    yaxis, yticks = altticks(ydata)

    cdict = {'red': ((0.01, 1, 0), (0.35, 0, 0), (0.66, 1, 1), (0.89,1, 1), (1, 0.5, 0.5)), \
         'green': ((0.01, 1, 0), (0.125,0, 0), (0.375,1, 1), (0.64,1, 1), (0.91,0,0), (1, 0, 0)), \
         'blue': ((0.01,1,0.5), (0.11, 1, 1), (0.34, 1, 1), (0.65,0, 0), (1, 0, 0))}
    
    my_cmap = colors.LinearSegmentedColormap('my_colormap',cdict,256)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ext = forceAspect(ax1,aspect = 1.0)
    im = ax1.imshow(data, vmin=0, vmax=1, cmap = my_cmap, extent = ex)
    fig.colorbar(im)
    ax1.set_xticklabels(xaxis,rotation=17)
    ax1.set_xticks(xticks)
    ax1.set_yticklabels(yaxis)
    ax1.set_yticks(yticks)
    plt.show()

    

def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    height = abs(extent[3]-extent[2])
    width = height*aspect
    newextent = [enextent[0],(extent[0]+width), extent[2],extent[3]]

    return newextent

def dateticks(axisdat,numticks = 10):
    l = len(axisdat)
    step = l//numticks
    ticklabels = axisdat[::step]
    tickmarks = range(0,l,step)

    ticklabels = [t.strftime('%H:%M') for t in ticklabels]

    return ticklabels[::-1], tickmarks

def altticks(axisdat, numticks = 10):
    l = len(axisdat)
    step = l//numticks
    ticklabels = axisdat[::step]
    tickmarks = range(0,l,step)
    ticklabels = [str(int(t)) for t in ticklabels]

    return ticklabels[::-1], tickmarks

if __name__ == '__main__':
    import pandas as pan
    import os
    os.chdir('C:/users/dashamstyr/Dropbox')
    df = pan.load('testdat.pickle')
    datetime = df.index
    alt = df.columns

    depol_plot(datetime, alt, df.T[::-1])
