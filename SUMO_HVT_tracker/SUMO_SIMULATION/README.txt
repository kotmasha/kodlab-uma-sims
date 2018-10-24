SUMO_SIMULATION
--------------------------------------------------------------------------------------
QUICKRUN
>>> python run.py environment.txt buildings.txt=OPTIONAL visibility.txt=OPTIONAL

run.py: contains all the necessary functions to output cPickled files of 
		'_name_.preamble' and '_name_.content'

environment.txt: contains environment variables to build the roads, edges, routes
				 and traffic in SUMO simulations. you can change number of grids,
				 dimension of blocks, input filename etc.

buildings.txt: contains the lower left and upper right coordinates of buildings

visibility.txt: contains the lower left and upper right coordinates of non-visible
				regions
--------------------------------------------------------------------------------------
EXPLANATION
run.py calls setup_network.py to build roads, edges, routes and traffic. All dependent
simulation *.xml files are build automatically by node.py, edg.py and route.py. 
setup_network.py also handles building network file *.net.xml automatically, and runs
the simulations with full output written to FILENAME_output.xml.

setup_network.py does NOT build any *.type.xml or *conn.xml which implements further
restrictions to the simulation. routes are fully randomized and there might be 
consecutive U-turns for a vehicle. 

Any simulation variable can be changed in environment.txt to build different models.
The cPickled output files will be created in the same directory with their specified
names ('_name_.preamble' and '_name_.content') and could be used in bastard.py
---------------------------------------------------------------------------------------
