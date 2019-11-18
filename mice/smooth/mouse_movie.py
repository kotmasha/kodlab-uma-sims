from __future__ import division
import sys
import os
import json
from cPickle import loads as unpickle
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation
from mouse_base import *
import time


def get_pickles(infile):
    for item in infile:
        yield json.loads(item)
    #try:
    #    while True:
    #        yield json.loads(infile.readline())
    #except EOFError:
    #    pass

def cx(ls):
    return complex(ls[0],ls[1])

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
supp_file_name=lambda i: os.path.join(DIRECTORY,RUN_NAME(i)+".sup")
input_file=open(input_file_name(NUM),'rb')
supp_file=open(supp_file_name(NUM),'rb')

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

# read the supplementary data file (a single json pickle)
SUPP=next(get_pickles(supp_file))

# form a random number generator using recorded initial state
RS=np.random.RandomState()
RS.set_state(unpickle(str(SUPP['seed'])))

# read the cheese parameters
CHEESE_PARAMS={
    'nibbles':preamble['nibbles'],
    'maxCheeses':int(SUPP['maxCheeses']),
    'Ncheeses':int(SUPP['Ncheeses']),
}

# read the mouse parameters
MOUSE_PARAMS={
    'horizon': np.float32(preamble['horizon']),
    'order': np.int32(preamble['order']),
    }

MOUSE_INIT={
    'pos':cx(SUPP['pos_out']),
    'pose':cx(SUPP['pose_out']),
    }
# read the initial list of cheese positions
CHEESE_LIST=[complex(p[0],p[1]) for p in SUPP['cheeseList']]

# call the arena constructor
arena = Arena_wmouse(
    xbounds=eval(preamble['xbounds']),
    ybounds=eval(preamble['ybounds']),
    cheese_params=CHEESE_PARAMS,
    cheese_list=CHEESE_LIST,
    mouse_params=MOUSE_PARAMS,
    mouse_init=MOUSE_INIT,
    visualQ=True,
    out_of_bounds=eval(preamble['out_of_bounds']),
    random_state=RS,
    )

# update data to be displayed in the movie:
arena.setMisc('counter',SUPP['counter'],lambda ar: ar.getMiscVal('counter')+1)

# function replaying the recorded activity of the mouse in the arena
def arena_advance(record):
    # prepare container for raw data
    DATA={}
    # obtain next record
    decode_mids(record,DATA)
    # update the arena according to new input
    # - we are assuming the mouse movement generates all changes
    arena.update(('teleport',(cx(DATA['pos_out']),cx(DATA['pose_out']))))

anim = animation.FuncAnimation(
    fig=arena.getRepn('global_view').getContent('global_fig'),
    func=arena_advance,
    #init_func=arena_init,
    frames=input,
    repeat=False,
    save_count=preamble['training_cycles']+preamble['run_cycles']+1,
    interval=10,
    )

while True:
    userInput=raw_input('Type anything to exit:\n')
    if userInput:
	exit(0)
#anim.save(os.path.join(DIRECTORY,RUN_NAME(NUM)+'.mp4'),dpi=100)#,writer='imagemagick')
