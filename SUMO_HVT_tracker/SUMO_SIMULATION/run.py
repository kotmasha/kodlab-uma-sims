import numpy as np 
import sys
import xml.etree.ElementTree as ET
import mobs
import math
from setup_network import setup
import cPickle

###
# environment variables
###

ENV_FILE = sys.argv[1]

# set_up network
setup(ENV_FILE)

ENVIRONMENT = {}

with open(ENV_FILE, 'r') as f:
	for line in f:
		line = line.strip()
		(key, val) = line.split("=")
		ENVIRONMENT[key] = val

X_GRID = int(ENVIRONMENT['X_GRID'])
Y_GRID = int(ENVIRONMENT['Y_GRID'])

BLOCK_DIM = int(ENVIRONMENT['BLOCK_DIM'])
DISC_PARA = int(ENVIRONMENT['DISC_PARA'])
LANE_WIDTH = int(ENVIRONMENT['LANE_WIDTH'])
FILENAME = ENVIRONMENT['FILENAME'] + '_output.xml'
CONTENT_FILE = ENVIRONMENT['FILENAME']+'.content'
PREAMBLE_FILE = ENVIRONMENT['FILENAME']+'.preamble'
TARGET_ASSIGNMENT = ENVIRONMENT['TARGET'],int(ENVIRONMENT['START_FRAME'])
TOTAL_FRAMES = int(ENVIRONMENT['END_TIME'])-int(ENVIRONMENT['BEGIN_TIME'])

# calculations done by ENV_VARIABLES
X_BOUND = BLOCK_DIM * (X_GRID)
Y_BOUND = BLOCK_DIM * (Y_GRID)

X_DISC = X_BOUND / DISC_PARA
Y_DISC = Y_BOUND / DISC_PARA

X_OFFSET = BLOCK_DIM * X_GRID / 2
Y_OFFSET = BLOCK_DIM * Y_GRID / 2

# Discretized 2D node from raw coordinates (x_pos,y_pos):
def node(x_pos,y_pos):
        return (x_pos-X_OFFSET)/DISC_PARA,(y_pos-Y_OFFSET)/DISC_PARA
        #return tmp.real,tmp.imag

# Discretized 2D vector from raw coordinates (x_val,y_val):
def vec(x_val,y_val):
        return mobs.icomplex(mobs.my_round(x_val / DISC_PARA),mobs.my_round(y_val / DISC_PARA))

# Discretized 2D vector from raw magnitude and angle 
def iexp(magnitude,angle):
        #turn into complex number:
        tmp=magnitude*np.exp(np.complex(0,(angle/180.)*np.pi))
        return vec(tmp.real,tmp.imag)

###
# MAP variables
# input: buildings.txt, visibility.txt
###
PAVEMENT = []
BUILDING_TEMP = []
OCCLUSIONS = []

if len(sys.argv) == 3:
	BUILDING = sys.argv[2]
	with open(BUILDING, 'r') as b:
		for line in b:
			line = line.strip()
			line = line.split(',')
			coor = [node(float(line[0]), float(line[1])), node(float(line[2]), float(line[3]))]
			BUILDING_TEMP.append(coor)

if len(sys.argv) == 4:
	BUILDING = sys.argv[2]
	OCCLUSIONS = sys.argv[3]

	with open(BUILDING, 'r') as b:
		for line in b:
			line = line.strip()
			line = line.split(',')
			coor = [node(float(line[0]), float(line[1])), node(float(line[2]), float(line[3]))]
			BUILDING_TEMP.append(coor)
	with open(OCCLUSIONS, 'r') as v:
		for line in v:
			line = line.strip()
			line = line.split(',')
			coor = [node(float(line[0]), float(line[1])), node(float(line[2]), float(line[3]))]
			OCCLUSIONS.append(coor)

### 
# XML parser 
# input: FILENAME_output.xml
# generate CONTENT data
# generate INFO data
###

TYPE = {}

tree = ET.parse(FILENAME)
p_two = open(CONTENT_FILE, 'wb')

root = tree.getroot()
for time_step in root:
	#t = time_step.attrib['timestep']
        CONTENT = {}
        for vehicle in time_step.find('vehicles'):
		car_id = vehicle.attrib['id']
		car_color = bin(int(car_id[-1]))[2:]
		car_x=float(vehicle.attrib['x'])
                car_y=float(vehicle.attrib['y'])
		#mobs_node = mobs.icomplex(node_value[0], node_value[1])
		car_speed = float(vehicle.attrib['speed'])
		car_angle = float(vehicle.attrib['angle'])
		car_velocity = iexp(car_speed,car_angle)
                if not(car_id in TYPE): 
		        TYPE[car_id] = car_color
                else:
                        pass
		CONTENT[car_id] = [vec(car_x,car_y), car_velocity]
        cPickle.dump(CONTENT, p_two, protocol=cPickle.HIGHEST_PROTOCOL)

        
###
# MAP DISCRETIZATION
###

##
# FIND ROAD NODES
##
X_LIST = [0, X_GRID]
Y_LIST = [0, Y_GRID]

boundary_nodes = []

# discretized nodes that contain roads
disc_nodes = []

getDiscNode = lambda some_list: [int((some_list[0] + LANE_WIDTH) / DISC_PARA), int((some_list[1] + LANE_WIDTH) / DISC_PARA)]

for y_ind, y_node in enumerate(Y_LIST):
	for x_ind, x_node in enumerate(X_LIST):
		node_x = x_node*BLOCK_DIM - LANE_WIDTH * math.pow(-1, x_ind)
		node_y = y_node*BLOCK_DIM - LANE_WIDTH * math.pow(-1, y_ind)
		boundary_nodes.append(getDiscNode([node_x, node_y]))

x_range = int(boundary_nodes[1][0] - boundary_nodes[0][0])
y_range = int(boundary_nodes[2][1] - boundary_nodes[0][1])

# go to up
for y_iter in xrange(Y_GRID+1):
	X_BL = int((boundary_nodes[0][0]) + y_iter * ((LANE_WIDTH + BLOCK_DIM) / DISC_PARA))
	X_TL = int((boundary_nodes[0][0] + 1) + y_iter * ((LANE_WIDTH + BLOCK_DIM) / DISC_PARA))
	for y_sweep in range(y_range + 1):
		disc_nodes.append([X_BL, y_sweep])
		disc_nodes.append([X_TL, y_sweep])

# go to left
for x_iter in xrange(X_GRID+1):
	Y_BL = int((boundary_nodes[0][1]) + x_iter * ((LANE_WIDTH + BLOCK_DIM) / DISC_PARA))
	Y_TL = int((boundary_nodes[0][1] + 1) + x_iter * ((LANE_WIDTH + BLOCK_DIM) / DISC_PARA))
	for x_sweep in range(x_range + 1):
		disc_nodes.append([x_sweep, Y_BL])
		disc_nodes.append([x_sweep, Y_TL])

##
# BUILD AND CLEAN UP BUILDING NODES
##
BUILDING_NODES = []
for coor in BUILDING_TEMP:
	x_nodes = [x for x in xrange(coor[0][0], coor[1][0]+1)]
	y_nodes = [y for y in xrange(coor[0][1], coor[1][1]+1)]
	for y_node in y_nodes:
		for x_node in x_nodes:
			if [x_node, y_node] not in disc_nodes:
				BUILDING_NODES.append([x_node, y_node])

##
# FIND VISIBILITY NODES
# OCC: WRITE THIS TO cPickle
##
occ = [[False]*X_DISC for i in xrange(Y_DISC)]

###
# SCENE: WRITE THIS TO cPickle
###
scene = [[1]*X_DISC for i in xrange(Y_DISC)]

for y in xrange(Y_DISC):
	for x in xrange(X_DISC):
		if [x,y] in disc_nodes:
			scene[y][x] = 0
		if [x,y] in PAVEMENT:
			scene[y][x] = 1
		if [x,y] in BUILDING_NODES:
			scene[y][x] = 2
		if [x,y] in OCCLUSIONS:
			occ[y][x] = True

p_one = open(PREAMBLE_FILE, 'wb')
cPickle.dump(scene, p_one, protocol=cPickle.HIGHEST_PROTOCOL)
cPickle.dump(occ, p_one, protocol=cPickle.HIGHEST_PROTOCOL)
cPickle.dump(TARGET_ASSIGNMENT, p_one, protocol=cPickle.HIGHEST_PROTOCOL)
cPickle.dump(TYPE, p_one, protocol=cPickle.HIGHEST_PROTOCOL)
cPickle.dump(TOTAL_FRAMES, p_one, protocol=cPickle.HIGHEST_PROTOCOL)
p_one.close()
p_two.close()
