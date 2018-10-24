import cPickle as nedosol
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


fin=open('out_plot.data','r')
fig=nedosol.load(fin)
fin.close()

plt.savefig('hello_plot.eps',dpi=300,format='eps')
plt.show()

