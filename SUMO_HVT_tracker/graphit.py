#from multiprocessing import Pool
import cPickle as nedosol
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
#from matplotlib.gridspec import GridSpec
#from mpl_toolkits.mplot3d import Axes3D


### Create a generator for unpickling a file of pickled objects
def get_pickles(infile):
    try:
        while True:
            yield nedosol.load(infile)
    except EOFError:
        pass

### Prepare figure
fig, ((ax_interval,ax_circle)) = plt.subplots(1,2,sharex='col',sharey='row')
fig.suptitle('TLE values over 1000 iterations',fontsize=14,fontweight='bold')
ax_interval.set_title('(a) Area-based TLE'); ax_interval.set_ylabel('TLE'); ax_interval.tick_params(labelsize=8);
ax_circle.set_title('(b) Distance-based TLE'); ax_circle.set_ylabel('TLE'); ax_circle.tick_params(labelsize=8); 


#print 'BLAH'

with open('hello.aout') as fin:
    # Prepare the data
    data=np.array([row for row in get_pickles(fin)]).T
    means=[np.mean(np.log(line)) for line in data]
    stdiv=[np.std(line) for line in data]

    # Prepare the plot
    t=range(len(data))
    ax_interval.plot(t,means,'b-')


with open('hello.dout') as fin:
    # Prepare the data
    data=np.array([row for row in get_pickles(fin)]).T
    means=[np.mean(np.log(line)) for line in data]
    stdiv=[np.std(line) for line in data]

    # Prepare the plot
    t=range(len(data))
    ax_circle.plot(t,means,'b-')


fout=open('out_plot.data','w')
nedosol.dump(fig,fout,nedosol.HIGHEST_PROTOCOL)
fout.close()
