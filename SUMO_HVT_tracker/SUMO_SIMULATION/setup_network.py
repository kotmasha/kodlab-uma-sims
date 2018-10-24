from node import build_nodes
from edg import build_edges
from route import build_routes
from xml.etree import ElementTree as ET
from random import randint
import sys
import os

def setup(ENV_FILE):
	### 
	#SETUP FILE FOR SIMULATION
	###
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
	FILENAME = ENVIRONMENT['FILENAME']
	NUMBER_OF_CARS = int(ENVIRONMENT['NUMBER_OF_CARS'])
	ROUTE_LENGTH = int(ENVIRONMENT['ROUTE_LENGTH'])
	BEGIN = ENVIRONMENT['BEGIN_TIME']
	END = ENVIRONMENT['END_TIME']

	###
	# build simulation
	###

	# file_names
	NODE_FILENAME = FILENAME + ".nod.xml"
	EDGE_FILENAME = FILENAME + ".edg.xml"
	NET_FILENAME = FILENAME + ".net.xml"
	ROUTE_FILENAME = FILENAME + ".rou.xml"

	# build .nod.xml, edg.xml, rou.xml
	nodes = build_nodes(X_GRID, Y_GRID, BLOCK_DIM, NODE_FILENAME)
	neighbors = build_edges(X_GRID, Y_GRID, EDGE_FILENAME)

	# build route_edges
	route_edges = {}

	for car in xrange(NUMBER_OF_CARS):
		# assign route id to the car
		route_id = ''.join(["route", str(car)])
		temp_edges = []

		# randomly pick a starting edge 
		PREV_NODE = str(randint(0, len(neighbors.keys())-1))

		for route in xrange(ROUTE_LENGTH):
			neighbor_list = neighbors[PREV_NODE]
			NEXT_NODE_INDEX = randint(0, len(neighbor_list)-1)
			NEXT_NODE = str(neighbor_list[NEXT_NODE_INDEX])
			ROUTE_EDGE = ''.join([PREV_NODE, "to", NEXT_NODE])
			temp_edges.append(ROUTE_EDGE)
			PREV_NODE = NEXT_NODE

		# convert route_edges[route_id] to a single string
		car_route = ' '.join(temp_edges)
		route_edges[route_id] = car_route

	build_routes(route_edges, NUMBER_OF_CARS, ROUTE_FILENAME)

	# build configuration FILE
	# their weird XML structure for .sumocfg makes everything
	# so difficult
	CONFIG_TEMPLATE_FILENAME = 'config_template.txt'
	CONFIG_FILENAME = FILENAME + '.sumocfg'

	lines = []
	with open(CONFIG_TEMPLATE_FILENAME, 'r') as config:
		while True:
			c = config.read(1)
			lines.append(c.lower())
			if not c:
				break
	temp_lines = [] 

	for index, char in enumerate(lines):
		temp_lines.append(char)
		if index - 10 > 0:
			check = ''.join(lines[index-9:index+1])
			if check == 'end value=':
				temp_lines.append('"'+END+'"')
		if index - 12 > 0:
			check = ''.join(lines[index-11:index+1])
			if check == 'begin value=':
				temp_lines.append('"'+BEGIN+'"')
		if index - 15 > 0:
			check = ''.join(lines[index-14:index+1])
			if check == 'net-file value=':
				temp_lines.append('"'+NET_FILENAME+'"')
		if index - 18 > 0:
			check = ''.join(lines[index-17:index+1])
			if check == 'route-files value=':
				temp_lines.append('"'+ROUTE_FILENAME+'"')
		
	with open(CONFIG_FILENAME, 'w') as config:
		write_line = ''.join(temp_lines)
		config.write(write_line)

	# modify edg.xml
	lines = ''
	# From keyword must change
	with open(EDGE_FILENAME, 'r') as edges:
		while True:
			c = edges.read(1)
			lines += c.lower()
			if not c:
				break

	with open(EDGE_FILENAME, 'w') as edges:
		edges.write(lines)

	# create NETWORK
	os.system('netconvert' + ' --node-files=' + NODE_FILENAME + ' --edge-files=' + EDGE_FILENAME + ' --output-file=' + NET_FILENAME)

	# run simulation
	OUTPUT_FILE = FILENAME + '_output.xml' 
	os.system('sumo -c ' + CONFIG_FILENAME + ' --full-output=' + OUTPUT_FILE)