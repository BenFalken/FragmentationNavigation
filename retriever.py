import gui, offline_navigation, online_navigation
from utilities import *
import pickle as pkl

def save_grid_cells(grid_cells):
	grid_cell_file = open('grid_cells', 'ab')
	pkl.dump(grid_cells, grid_cell_file)
	grid_cell_file.close()

def save_environment(environment):
	map_file = open('environment', 'ab')
	pkl.dump(environment, map_file)
	map_file.close()

def save_all_sensory_maps(all_sensory_maps):
	all_sensory_maps_file = open('all_sensory_maps', 'ab')
	pkl.dump(all_sensory_maps, all_sensory_maps_file)
	all_sensory_maps_file.close()

def save_surprise_map(surprise_map):
	surprise_map_file = open('surprise_map', 'ab')
	pkl.dump(surprise_map, surprise_map_file)
	surprise_map_file.close()

def save_distance_matrix(distance_matrix):
	distance_matrix_file = open('distance_matrix', 'ab')
	pkl.dump(distance_matrix, distance_matrix_file)
	distance_matrix_file.close()

def save_isomap(isomap):
	isomap_file = open('isomap', 'ab')
	pkl.dump(isomap, isomap_file)
	isomap_file.close()

def retrieve_grid_cells():
	try:
		grid_cell_file = open('grid_cells','rb')
		grid_cells = pkl.load(grid_cell_file)
		grid_cell_file.close()
		return grid_cells
	except:
		grid_cells = online_navigation.create_all_grid_cells()
		save_grid_cells(grid_cells)
		return grid_cells

def retrieve_environment():
	try:
		map_file = open('environment','rb')
		all_data = pkl.load(map_file)
		environment = all_data['environment']
		[x_pos, y_pos] = all_data["pos"]
		map_file.close()
		return environment, x_pos, y_pos
	except:
		environment, x_pos, y_pos = gui.draw_map()
		save_environment({"environment": environment, "pos": [x_pos, y_pos]})
		return environment, x_pos, y_pos

def retrieve_all_sensory_maps(environment):
	try:
		all_sensory_maps_file = open('all_sensory_maps','rb')
		all_sensory_maps = pkl.load(all_sensory_maps_file)
		all_sensory_maps_file.close()
		return all_sensory_maps
	except:
		print("We will now create the sensory map file. Please stand by.")
		all_sensory_maps = offline_navigation.generate_sensory_map_for_all_points(environment)
		save_all_sensory_maps(all_sensory_maps)
		return all_sensory_maps

def retrieve_surprise_map(all_sensory_maps):
	try:
		surprise_map_file = open('surprise_map','rb')
		surprise_map = pkl.load(surprise_map_file)
		surprise_map_file.close()
		return surprise_map
	except:
		print("We will now create the surprise map file. Please stand by.")
		surprise_map = offline_navigation.generate_surprise_map(all_sensory_maps)
		save_surprise_map(surprise_map)
		return surprise_map

def retrieve_isomap(distance_matrix):
	try:
		isomap_file = open('isomap','rb')
		isomap = pkl.load(isomap_file)
		isomap_file.close()
		return isomap["isomap"], isomap["labels"]
	except:
		print("We will now create the isomap file. Please stand by.")
		isomap, labels = offline_navigation.make_isomap(distance_matrix)
		save_isomap({"isomap": isomap, "labels": labels})
		return isomap, labels

def retrieve_distance_matrix(surprise_map):
	try:
		distance_matrix_file = open('distance_matrix','rb')
		distance_matrix = pkl.load(distance_matrix_file)
		distance_matrix_file.close()
		return distance_matrix
	except:
		distance_matrix = offline_navigation.make_distance_matrix(surprise_map)
		return distance_matrix