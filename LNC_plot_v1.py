def depol_plot(xdata, ydata, data):
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors

    #set date(s) for plot title
    days = list(set(t.strftime('%B %d, %Y') for t in xdata))
    sdays = list(set(t.strftime('%Y%m%d') for t in xdata))
    
    if len(days) == 1:
        titledays = days[0]
        savedays = sdays[0]
    else:
        titledays = '%s - %s'%(days[0],days[-1])
        savedays = '%s-%s'%(sdays[0],sdays[-1])

    title = 'Depolarization Ratio: %s'%titledays
    savetitle = 'Depolrat_%s'%savedays
    
    #set colormap to be the same as 'jet' with the addition of white color for
    #depol ratios set to identiacally zero because they couldn't be calculated
    cdict = {'red': ((0,1,1),
                     (0.0001, 1, 0),
                     (0.35, 0, 0),
                     (0.66, 1, 1),
                     (0.89,1, 1),
                     (1, 0.5, 0.5)),
         'green': ((0,1,1),
                   (0.0001, 1, 0),
                   (0.125,0, 0),
                   (0.375,1, 1),
                   (0.64,1, 1),
                   (0.91,0,0),
                   (1, 0, 0)),
         'blue': ((0,1,1),
                  (0.0001,1,0.5),
                  (0.11, 1, 1),
                  (0.34, 1, 1),
                  (0.65,0, 0),
                  (1, 0, 0))}
    
    my_cmap = colors.LinearSegmentedColormap('my_colormap',cdict,1064)
    
    #create figure and plot image of depolarization ratios
    font = 18
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    im = ax1.imshow(data, vmin=0, vmax=1, cmap = my_cmap)
    forceAspect(ax1,2.0)
    t = ax1.set_title(title, fontsize = font+10)
    t.set_y(1.1)
    plt.subplots_adjust(top = 0.86, bottom = 0.01, left = 0.09, right = 0.95)
    
    cbar = fig.colorbar(im, orientation = 'horizontal', pad = 0.15, aspect = 40)
    cbar.ax.tick_params(labelsize = 18)

    #set axis ranges and tickmarks based on data ranges
    dateticks(ax1, xdata)
    ax1.set_xlabel('Time [PDT]',fontsize = font+4, labelpad = 15)

    for line in ax1.xaxis.get_ticklines():
        line.set_markersize(10)
        line.set_markeredgewidth(1)
        
    altticks(ax1, ydata)
    ax1.set_ylabel('Altitude [m]', fontsize = font+4, labelpad = 15)

    for line in ax1.yaxis.get_ticklines():
        line.set_markersize(10)
        line.set_markeredgewidth(1)
        
    ax1.axis('tight')
    fig.set_size_inches(20,10)
    plt.savefig(savetitle,dpi = 100, edgecolor = 'b', bbox_inches = 'tight')
    plt.show()

    

def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)

def dateticks(ax, axisdat,numticks = 10, font = 18):
    import matplotlib.pyplot as plt
    l = len(axisdat)
    step = l//numticks
    ticklabels = axisdat[::step]
    tickmarks = range(0,l,step)

    ticklabels = [t.strftime('%H:%M') for t in ticklabels]
    
    plt.xticks(tickmarks, ticklabels, fontsize = font)

def altticks(ax, axisdat, numticks = 10, font = 18):
    import matplotlib.pyplot as plt
    
    numpoints = len(axisdat)
    step = numpoints//numticks
    locstep = 10//numticks
    ticklabels = axisdat[::step]
    tickmarks = range(0,numpoints,step)
    ticklocs = range(0,10,locstep)
    ticklabels = [str(int(t)) for t in ticklabels]

    plt.yticks(tickmarks[:-1],ticklabels[::-1], fontsize = font)


if __name__ == '__main__':
    import pandas as pan
    import os
    os.chdir('C:/users/User/Dropbox')
    df = pan.load('testdat.pickle')
    datetime = df.index
    alt = df.columns

    depol_plot(datetime, alt, df.T[::-1])
