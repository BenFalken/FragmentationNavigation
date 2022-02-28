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

# In order to compare maps that are offset from each other, you must convert to cartesian coordinates,
# Then you must center both at the origin, then convert both back into polar coordinates
def offset_and_shift_map_at_origin(offset_x, offset_y, map_of_environment):
	# Initialize the sensory map that shall be shifted to approximate the current "investigating map" at the offset point
	map_at_origin_shifted_to_fit_investigating_map = np.zeros((TICKS))
	# For point in the shifted map, find an approximation in terms of valid values r, theta
	for tick in range(TICKS):
		x_point = map_at_origin[tick]*X_COMP[tick] + offset_x
		y_point = map_at_origin[tick]*Y_COMP[tick] - offset_y
		# Theta values are discretized
		#So converting cartesian points to polar coordinates means we have to adjust them to fit the discrete theta values we outlined in constants.py
		round_theta_index = get_round_theta_index(x_point, y_point)
		rad = np.sqrt(x_point**2 + y_point**2)
		map_at_origin_shifted_to_fit_investigating_map[round_theta_index] = np.round(rad, 1)
	return map_at_origin_shifted_to_fit_investigating_map

def find_error_between_maps(offset_x, offset_y, map_at_origin, investigating_map, offset_or_gaussian="offset"):
	map_at_origin_shifted_to_fit_investigating_map = offset_and_shift_map_at_origin(offset_x, offset_y, map_at_origin)

	# This is a container for filling the unfilled polar coordinates at the acceptable theta ticks
	new_rad_storage = np.zeros(TICKS)

	for tick in range(TICKS):
		rad = map_at_origin_shifted_to_fit_investigating_map[tick]
		# If one of the slots at a value theta is empty, we really ought to fill it
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

	# Choose whether to evaluate "surprisal" by how unpredictable the radius of the sensory map at a certain value might be,
	# or rather by how much the two maps differ from each other
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
			if np.sum(all_sensory_maps[y][x]) > 0: #and edge:
				offset_x = (col - x)*STEP
				offset_y = (row - y)*STEP
				surprise = find_error_between_maps(offset_x, offset_y, all_sensory_maps[row][col], all_sensory_maps[y][x])
				surprise_map.append(surprise)
	
	if len(surprise_map) > 0:
		return sum(surprise_map) / len(surprise_map)
	else:
		return 0

# For testing purposes only. Find the surprisal at all points on a map based off of one sigle reference
def generate_surprise_for_one_point_fixed(row, col, all_sensory_maps):
	row = int(row/STEP)
	col = int(col/STEP)
	surprise_map = np.zeros(all_sensory_maps.shape[:2])
	for y in range(0, all_sensory_maps.shape[0]):
		for x in range(0, all_sensory_maps.shape[1]):
			print(y, x)
			if np.sum(all_sensory_maps[y][x]) > 0:
				offset_x = (col - x)*STEP
				offset_y = (row - y)*STEP
				surprise = np.linalg.norm(all_sensory_maps[row][col] - all_sensory_maps[y][x])
				surprise = find_error_between_maps(offset_x, offset_y, all_sensory_maps[row][col], all_sensory_maps[y][x])
				surprise_map[y][x] = surprise
	plt.imshow(surprise_map)
	plt.show()
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
			if surprise_map[i]["surprise"] != surprise_map[j]["surprise"]:
				# We multiply the magnitude of the distance vectore by the prximity of the two points being compared
				distance_matrix[i][j] = np.log(np.abs(surprise_map[i]["surprise"] - surprise_map[j]["surprise"]))*np.sqrt((surprise_map[i]["coord"][0] - surprise_map[j]["coord"][0])**2 +(surprise_map[i]["coord"][1] - surprise_map[j]["coord"][1])**2)
			else:
				distance_matrix[i][j] = 0
	return distance_matrix

# Create isomap from distance matrix of all points. Apply dbscan clustering 
def make_isomap(distance_matrix):
	# Create isomap
	mds = MDS(n_components=2, dissimilarity="precomputed", random_state=6)
	results = mds.fit(distance_matrix)
	coords = results.embedding_
	# Cluster resulting points
	clustering = DBSCAN(eps=EPS, min_samples=2).fit(coords)
	labels = clustering.labels_
	return coords, labels

# Go through all clusters and cut out the ones with less than the minimum required points to be considered a unique region
def determine_significance_of_each_region(labels):
	each_room_total_point_number = {}
	for fixed_label in labels:
		cluster_point_count = 0
		for label in labels:
			if label == fixed_label:
				cluster_point_count += 1
		each_room_total_point_number[str(fixed_label)] = cluster_point_count
	return each_room_total_point_number
		
# Determine the number of regions with a significant number of points (ie: actual rooms)
def determine_number_of_significant_regions(each_room_total_point_number):
	unique_region_count = 0
	for key in each_room_total_point_number.keys():
		point_number = each_room_total_point_number[key]
		if point_number > MIN_POINTS_FOR_CLUSTER_TO_BE_CONSIDERED_REGION:
			unique_region_count += 1
	return unique_region_count

# Don't bother adding the points that belong to the regions we don't care about
def clean_coords_of_nonsignificant_regions(coords, labels, each_room_total_point_number):
	coords = np.array([coords[i] for i in range(coords.shape[0]) if each_room_total_point_number[str(labels[i])] > MIN_POINTS_FOR_CLUSTER_TO_BE_CONSIDERED_REGION])
	return coords

# Plot isomap in matplotlib
def present_isomap(coords, labels, show_clustering):
	if show_clustering:
		each_room_total_point_number = determine_significance_of_each_region(labels)
		unique_region_count = determine_number_of_significant_regions(each_room_total_point_number)
		coords = clean_coords_of_nonsignificant_regions(coords, labels, each_room_total_point_number)
		colors = [(randint(0, 255)/255, randint(0, 255)/255, randint(0, 255)/255) for _ in range(labels.size + 1)]
		colors_for_each_coord = [colors[labels[i]] for i in range(labels.size) if each_room_total_point_number[str(labels[i])] > MIN_POINTS_FOR_CLUSTER_TO_BE_CONSIDERED_REGION]
		plt.title("There are " + str(unique_region_count) + " regions in the environment given")
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
	#generate_surprise_for_one_point_fixed(50, 400, all_sensory_maps)
	surprise_map = retriever.retrieve_surprise_map(all_sensory_maps)
	distance_matrix = retriever.retrieve_distance_matrix(surprise_map)
	isomap, labels = retriever.retrieve_isomap(distance_matrix)
	# Graph everything
	present_surprise_map(surprise_map)
	present_isomap(isomap, labels, show_clustering=True)
	



