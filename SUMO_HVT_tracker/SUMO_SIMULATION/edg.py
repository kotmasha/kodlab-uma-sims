from xml.etree.ElementTree import Element, ElementTree

def build_edges(X_GRID, Y_GRID, FILENAME):
	edges = Element('edges')

	edge_list = []
	neighbors = {}

	prop = lambda initial, terminal: Element('edge', From=initial, id=''.join([initial, "to", terminal]), to=terminal)
	for y in xrange(Y_GRID):
		for x in xrange(X_GRID):

			left = x + 1
			right = x - 1
			up = y + 1
			down = y - 1

			initial = str(x + y * X_GRID)
			neighbors[initial] = []

			if left < X_GRID:
				terminal = str(left + y * X_GRID)
				edge_list.append(prop(initial, terminal))
				neighbors[initial].append(terminal)

			if right > -1:
				terminal = str(right + y * X_GRID)
				edge_list.append(prop(initial, terminal))
				neighbors[initial].append(terminal)

			if up < Y_GRID:
				terminal = str(x + up * X_GRID)
				edge_list.append(prop(initial, terminal))
				neighbors[initial].append(terminal)

			if down > -1:
				edge_to = str(x + down * X_GRID)
				edge_list.append(prop(initial, terminal))
				neighbors[initial].append(terminal)

	edges.extend(edge_list)

	tree = ElementTree(edges)

	tree.write(FILENAME, xml_declaration=True, 
			encoding="UTF-8", method="xml")

	return neighbors
