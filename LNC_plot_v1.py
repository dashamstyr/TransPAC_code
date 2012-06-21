def plot_test(data):
    import matplotlib.pyplot as plt
    import pandas as pan

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    im = ax1.imshow(data, vmin=0, vmax=1)
    fig.colorbar(im)
    forceAspect(ax1,aspect = 1.0)
    
    plt.show()

def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)
