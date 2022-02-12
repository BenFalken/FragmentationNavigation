import numpy as np
from constants import *

# Determines if the tracker used to generate the spatial map is about to exit bounds
def in_bounds(val):
	return val < SIDE_LEN and val >= 0

# Clamps value at a maximum and minumum value
def clamp(val, max_val):
	return min(max(val, 0), max_val)

# Generates sensory map at a reference point (x, y)
def generate_sensory_map(x, y, environment):
	sensory_map = np.zeros((TICKS))
	for tick in range(TICKS):
		map_in_bounds = True
		edge_reached = False
		# Increment tracker until boundary reached
		t = 0
		while map_in_bounds and not edge_reached:
			map_in_bounds = (in_bounds(x + t*X_COMP[tick]) and in_bounds(y - t*Y_COMP[tick]))
			edge_reached = environment[int(y - t*Y_COMP[tick])][int(x + t*X_COMP[tick])] == 1
			t += dt
		sensory_map[tick] = t
	return sensory_map

# Given a cartesian point, find an approximate in polar coordinate theta value, where theta is a member of THETAS
def get_round_theta_index(x, y):
	if x >= 0:
		theta = np.arctan(y/x)
	elif y < 0:
		theta = np.arctan(y/x)  - np.pi
	elif y > 0:
		theta = -1*np.arctan(y/x)  + np.pi/2
	else:
		theta = 0
	round_theta_index = int( clamp( np.round( (np.round( TICKS*theta/np.pi ) + TICKS )/2 ), max_val=TICKS - 1 ) )
	return round_theta_index