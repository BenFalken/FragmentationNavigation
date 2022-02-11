import retriever
import numpy as np
import matplotlib.pyplot as plt
from random import randint
from constants import *
from utilities import *

from scipy.stats import multivariate_normal
from sklearn.manifold import MDS
from sklearn.cluster import DBSCAN

# Iterates through all rows and columns of a map and generates a map at each respective point
def generate_sensory_map_for_all_points(environment):
	all_sensory_maps = np.zeros((int(SIDE_LEN/STEP), int(SIDE_LEN/STEP), TICKS))
	for i in range(0, SIDE_LEN, STEP):
		row = int(i/STEP)
		for j in range(0, SIDE_LEN, STEP):
			col = int(j/STEP)
			if environment[i][j] != 1:
				sensory_map = generate_sensory_map(j, i, environment)
				all_sensory_maps[row][col] = sensory_map
	return all_sensory_maps

# Construct gaussian matrix from the real values of the sensory map, and evaluate by the estimates values
def calculate_error_by_gaussian(est_vals, real_vals):
	# Construct gaussian
	mean_real = np.array([np.mean(real_vals), np.mean(real_vals)])
	cov_real = np.cov(real_vals, real_vals)[0][0]
	cov_real = np.array([[cov_real, 0], [0, cov_real]])
	distr_real = multivariate_normal(cov = cov_real, mean = mean_real)
	# Evaluate matrix at each position tick
	surprise = 0
	for tick in range(TICKS):
		surprise += distr_est.pdf((est_vals[tick], real_vals[tick]))
	return surprise

def find_error_between_maps(offset_x, offset_y, map_at_origin, investigating_map, offset_or_gaussian="offset"):
	# Initialize the sensory map that shall be shifted to approximate the current "investigating map" at the offset point
	map_at_origin_shifted_to_fit_investigating_map = np.zeros((TICKS))
	# These are the true cartesian points of the shifted sensory map, 
	shifted_origin_map_points_x, shifted_origin_map_points_y = [], []
	approx_shifted_origin_map_points_x, approx_shifted_origin_map_points_y = [], []
	investigating_map_points_x, investigating_map_points_y = [], []
	# For point in the shifted map, find an approximation in terms of valid values r, theta
	for tick in range(TICKS):
		x_point = map_at_origin[tick]*X_COMP[tick] + offset_x
		y_point = map_at_origin[tick]*Y_COMP[tick] - offset_y
		round_theta_index = get_round_theta(x_point, y_point)
		
		rad = np.sqrt(x_point**2 + y_point**2)
		map_at_origin_shifted_to_fit_investigating_map[round_theta_index] = np.round(rad, 1)

		shifted_origin_map_points_x.append(x_point)
		shifted_origin_map_points_y.append(y_point)

		investigating_map_points_x.append(investigating_map[tick]*X_COMP[tick])
		investigating_map_points_y.append(investigating_map[tick]*Y_COMP[tick])

	# You be storing the new rads here
	new_rad_storage = np.zeros(TICKS)

	for tick in range(TICKS):
		rad = map_at_origin_shifted_to_fit_investigating_map[tick]
		# If the algorithm fucked up and there's no value stored at a particular value of theta
		if tick > 0 and tick < TICKS and rad == 0:
			# Consatantly expand the range at which you look to find the nearest points to make a mean from
			num_valids = np.argwhere(map_at_origin_shifted_to_fit_investigating_map[tick - 1: tick + 2] > 0).size
			counter = 0
			while num_valids == 0:
				counter += 1
				num_valids = np.argwhere(map_at_origin_shifted_to_fit_investigating_map[tick - (2 + counter): tick + (3 + counter)] > 0).size
			# Make a new radius that's basically an approximation from other surrounding values	
			new_rad = np.sum(map_at_origin_shifted_to_fit_investigating_map[tick - (2 + counter): tick + (3 + counter)])/num_valids
			new_rad_storage[tick] = new_rad

	for tick in range(TICKS):
		if new_rad_storage[tick] != 0:
			map_at_origin_shifted_to_fit_investigating_map[tick] = np.round(new_rad_storage[tick], 1)

	if offset_or_gaussian == "offset":
		error = np.linalg.norm(map_at_origin_shifted_to_fit_investigating_map - investigating_map)
	else:
		error = calculate_error_by_gaussian(map_at_origin_shifted_to_fit_investigating_map, investigating_map)
	return error

# For every point where sensory maps were recorded, find the surprisal at that point
def generate_surprise_map(all_sensory_maps):
	surprise_map = np.zeros(all_sensory_maps.shape[:2])
	for row in range(surprise_map.shape[0]):
		for col in range(surprise_map.shape[1]):
			if np.sum(all_sensory_maps[row][col]) != 0:
				surprise_for_one_point = generate_surprise_for_one_point(row, col, all_sensory_maps)
				surprise_map[row][col] = surprise_for_one_point
	return surprise_map

# Find points at a fixed radius away from the reference. Generate the average surprisal by summing all surprises
def generate_surprise_for_one_point(row, col, all_sensory_maps):
	surprise_map = []
	for y in range(clamp(row - RADIUS, all_sensory_maps.shape[0]), clamp(row + RADIUS, all_sensory_maps.shape[0])):
		for x in range(clamp(col - RADIUS, all_sensory_maps.shape[1]), clamp(col + RADIUS, all_sensory_maps.shape[1])):
			# Test if current offset point lies on the circle at RADIUS radius away from row, col
			edge = np.abs(np.sqrt((col - x)**2 + (row - y)**2) - RADIUS) < ERROR
			if np.sum(all_sensory_maps[y][x]) > 0 and edge:
				offset_x = (col - x)*STEP
				offset_y = (row - y)*STEP
				surprise = find_error_between_maps(offset_x, offset_y, all_sensory_maps[row][col], all_sensory_maps[y][x])
				surprise_map.append(surprise)
	return sum(surprise_map) / len(surprise_map)

# For testing purposes only. Find the surprisal at all points on a map based off of one sigle reference
def generate_surprise_for_one_point_fixed(row, col, all_sensory_maps):
	row = int(row/STEP)
	col = int(col/STEP)
	surprise_map = np.zeros(all_sensory_maps.shape[:2])
	for y in range(0, all_sensory_maps.shape[0]):
		for x in range(0, all_sensory_maps.shape[1]):
			if np.sum(all_sensory_maps[y][x]) > 0:
				offset_x = (col - x)*STEP
				offset_y = (row - y)*STEP
				surprise = find_error_between_maps(offset_y, offset_x, all_sensory_maps[row][col], all_sensory_maps[y][x])
				surprise_map[y][x] = surprise
	return surprise_map

# Calculate the distance matrox matrix off all the points with surprisal scores
def make_distance_matrix(surprise_map):
	new_surprise_map = []
	for i in range(surprise_map.shape[0]):
		for j in range(surprise_map.shape[1]):
			if surprise_map[i][j] != 0:
				new_surprise_map.append({"surprise": surprise_map[i][j], "coord":  (i, j)}) 
	surprise_map = np.array(new_surprise_map)
	distance_matrix = np.zeros((surprise_map.size, surprise_map.size))
	for i in range(surprise_map.size):
		for j in range(surprise_map.size):
			if surprise_map[i]["surprise"] != surprise_map[j]["surprise"]:\
				# We multiply the magnitude of the distance vectore by the prximity of the two points being compared
				distance_matrix[i][j] = np.log(np.abs(surprise_map[i]["surprise"] - surprise_map[j]["surprise"]))*np.sqrt((surprise_map[i]["coord"][0] - surprise_map[j]["coord"][0])**2 +(surprise_map[i]["coord"][1] - surprise_map[j]["coord"][1])**2)
			else:
				distance_matrix[i][j] = 0
	return distance_matrix

# Create isomap from distance matrix of all points. Apply dbscan clustering 
def create_isomap(distance_matrix):
	# Create isomap
	mds = MDS(n_components=2, dissimilarity="precomputed", random_state=6)
	results = mds.fit(distance_matrix)
	coords = results.embedding_
	# Cluster resulting points
	clustering = DBSCAN(eps=EPS, min_samples=2).fit(coords)
	labels = clustering.labels_
	return coords, labels

# Plot isomap in matplotlib
def present_isomap(coords, labels, show_clustering):
	if show_clustering:
		num_room_types = max([labels[i] for i in range(labels.size)]) + 1
		colors = [(randint(0, 255)/255, randint(0, 255)/255, randint(0, 255)/255) for _ in range(num_room_types)]
		colors_for_each_coord = [colors[labels[i]] for i in range(labels.size)]
		plt.title("There are " + str(num_room_types) + " rooms in the environment given")
	else:
		colors_for_each_coord = [(100, 100, 255) for _ in range(coords.shape[0])]
		plt.title("Approximate isomap of environment below")
	plt.scatter(
	    coords[:, 0], coords[:, 1], marker = 'o', color=colors_for_each_coord
	)
	plt.show()

# Plot the surprise map in matplotlib
def present_surprise_map(surprise_map):
	plt.title("Surprisal map of all positions in environment below")
	plt.imshow(surprise_map)
	plt.show()

# Main subroutine
def run_navigation():
	# Fetch all necessary data
	environment, pos_x, pos_y = retriever.retrieve_environment()
	all_sensory_maps = retriever.retrieve_all_sensory_maps(environment)
	surprise_map = retriever.retrieve_surprise_map(all_sensory_maps)
	distance_matrix = retriever.retrieve_distance_matrix(surprise_map)
	isomap, labels = retriever.retrieve_isomap(distance_matrix)
	# Graph everything
	present_surprise_map(surprise_map)
	present_isomap(isomap, labels, show_clustering=True)



