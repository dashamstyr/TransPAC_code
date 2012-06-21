import matplotlib.pyplot as plt
import pandas as pan

testdat = pan.load('testdat.pickle')

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.pcolor(testdat)

plt.show()
