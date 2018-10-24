from __future__ import division
import sys
import os
import json
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation
from mouse_base import *
from numpy.random import randint as rnd
import time

#Writer = animation.writers['ffmpeg']
#writer = Writer(fps=2,bitrate=10000)#, metadata=dict(artist='Me'), bitrate=1800)

def get_pickles(infile):
    for item in infile:
        yield json.loads(item)
    #try:
    #    while True:
    #        yield json.loads(infile.readline())
    #except EOFError:
    #    pass

def icx_pair(ls):
    return icomplex(ls[0],ls[1])

NAME=sys.argv[1]
NUM=int(sys.argv[2])
DIRECTORY=os.path.join(os.path.abspath('.'),NAME)

# load the preamble file
preamble_file_name=os.path.join(DIRECTORY,NAME+'.pre')
preamblef=open(preamble_file_name,'rb')
preamble=json.load(preamblef)
preamblef.close()

# select input file according to provided run number
RUN_NAME=lambda i: NAME+"_"+str(i)
input_file_name=lambda i: os.path.join(DIRECTORY,RUN_NAME(i)+".dat")
input_file=open(input_file_name(NUM),'rb')
# prepare generator object unpickling the data file
input = get_pickles(input_file)

### data decoder adds record to database
def decode_mids(rec,data):
    for mid,item in zip(preamble['mids_recorded'],rec['mids_recorded']):
        #del data[mid]
        data[mid]=item
        #data[mid].append(item)


### initialize an arena to be drawn in the movie
#

# call the arena constructor
arena = Arena_base(
    eval(preamble['xbounds']),
    eval(preamble['ybounds']),
    eval(preamble['out_of_bounds'])
    )

cheeseParams={
    'nibbles':preamble['nibbles'],
    'nibbleDist':preamble['nibbleDist'],
    }


# prepare container for raw data
DATA={}
# decode first frame to initialize the arena 
decode_mids(next(input),DATA)
# place cheeses in arena
for objtag in DATA['che_out']:
    arena.addCheese(
        str(objtag),
        icx_pair(DATA['che_out'][str(objtag)]),
        cheeseParams,
        )
# place mouse in arena
arena.addMouse(
    'mus',
    icx_pair(DATA['pos_out']),
    {'viewSize':preamble['viewportSize'],'direction' : icx_pair(DATA['dir_out'])},
    )
# update data to be displayed in the movie:
arena.putmisc(str(DATA['counter']))
arena.generateHeatmap('elevation')
#arena.getMice('mus').generateHeatmap('elevation')

# function replaying the recorded activity of the mouse in the arena
def animate(record):
    # obtain next record
    decode_mids(record,DATA)
    # update the arena according to new input
    arena.putmisc(str(DATA['counter']))
    flag=False
    for objtag in arena._objects.keys():
        obj=arena._objects[objtag]
        if obj._type=='mouse':
            # teleport the mouse to its new position
            obj.teleport(icx_pair(DATA['pos_out']),icx_pair(DATA['dir_out']))
        elif obj._type=='cheese':
            # remove the cheese if it is gone
            if not objtag in DATA['che_out']:
                obj.remove()
                flag=True
            else:
                pass
        else:
            pass
    # update the plot
    if flag:
        arena.updateHeatmapFull('elevation')
    else:
        arena.updateHeatmap('elevation')
    #arena.getMice('mus').updateHeatmapFull('elevation')

anim = animation.FuncAnimation(
    fig=arena._fig,
    func=animate,
    #init_func=animation_init,
    frames=input,
    repeat=False,
    save_count=preamble['total_cycles'],
    interval=50,
    )

plt.show()
#anim.save(os.path.join(DIRECTORY,RUN_NAME(NUM)+'.mp4'),dpi=100)#,writer='imagemagick')
