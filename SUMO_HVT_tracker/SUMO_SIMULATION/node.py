from xml.etree.ElementTree import Element, ElementTree

def build_nodes(X_GRID, Y_GRID, BLOCK_DIM, FILENAME):
	nodes = Element('nodes')

	node_list = []

	X_START = -(BLOCK_DIM * int(X_GRID / 2))
	Y_START = -(BLOCK_DIM * int(Y_GRID / 2))

	# CHANGE THIS LATER
	# for node in xrange((X_GRID+1)*(Y_GRID+1)):
	for y_node in xrange(Y_GRID):
		for x_node in xrange(X_GRID):

			# CHANGE THIS TO str(node)
			node_id = str(x_node + y_node * X_GRID)

			# need to UPDATE this
			node_x =  str(X_START + x_node*BLOCK_DIM)
			node_y = str(Y_START + y_node*BLOCK_DIM)
			node_list.append(Element('node', id=node_id, x=node_x, y=node_y)) 

	nodes.extend(node_list)

	tree = ElementTree(nodes)
	tree.write(FILENAME, xml_declaration=True, 
			encoding="UTF-8", method="xml")
	return node_list