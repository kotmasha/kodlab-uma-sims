from xml.etree.ElementTree import Element, ElementTree

def build_routes(ROUTE_EDGES, NUMBER_OF_CARS, FILENAME):
	routes = Element('routes')

	type_list = []

	# add vType
	# vehicle type is STANDARD Krauss vehicle type
	type_list.append(Element('vType', id="type1", accel="0.8", decel="4.5", 
		sigma="0.5", length="5", maxSpeed="70"))

	for key, value in ROUTE_EDGES.items():
		type_list.append(Element('route', id=key, edges=value))

	for car in xrange(NUMBER_OF_CARS):
		car_id = "veh" + str(car)
		route_id = "route" + str(car)
		depart_time = str(0)

		# color assignment
		color = [0, 0, 0]
		c_index = (car % 3)
		c_inc = int(car / 3) + 1
		color[c_index] = c_inc
		c_string = ','.join(str(e) for e in color)
		type_list.append(Element('vehicle', depart=depart_time, id=car_id, 
			route=route_id, type="type1", color=c_string))

	routes.extend(type_list)

	tree = ElementTree(routes)
	tree.write(FILENAME, xml_declaration=True, 
			encoding="UTF-8", method="xml")