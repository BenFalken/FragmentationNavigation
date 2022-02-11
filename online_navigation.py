import gui, retriever
import numpy as np
import matplotlib.pyplot as plt
from random import randint, choice
from constants import *
from utilities import *

# Make a cartesian array of BVCs and find the frequencies of each one given the current bounds in terms of r, theta
def update_bvc(x, y, environment):
	bvc_cartesian_array = np.zeros((SIDE_LEN, SIDE_LEN))
	sensory_map = generate_sensory_map(x, y, environment)
	for bvc_r in range(0, BVC_RADIUS, dt):
		for bvc_theta_index in range(TICKS):
			bvc_theta = THETAS[bvc_theta_index]
			bvc_x = int(x + bvc_r*X_COMP[tick])
			bvc_y = int(y - bvc_r*Y_COMP[tick])
			bvc_cartesian_array[bvc_x][bvc_y] = get_total_freq(sensory_map, bvc_r, bvc_theta)
	return bvc_cartesian_array

# Integrate all values of r and theta across each BVC
def get_total_freq(sensory_map, preferred_rad, preferred_theta):
	total_freq = 0
	for tick in TICKS:
		theta = THETAS[tick]
		rad = sensory_map[tick]
		total_freq += freq_at_r_theta(rad, theta, preferred_rad, preferred_theta, sigma_angle=1):
	return total_freq

# Get the frequency contribution to a BVC at each distinct r and theta
def freq_at_r_theta(r, theta, d, phi, sigma_angle):
	sigma_rad = d
	r_component = np.exp((-1*(r-d)**2)/(2*sigma_rad**2))/np.sqrt(2*pi*sigma_rad**2)
	theta_component = np.exp((-1*(theta-phi)**2)/(2*sigma_angle**2))/np.sqrt(2*pi*sigma_angle**2)
	return r_component*theta_component

# Chooses a new velocity weighted by previous velocity. This function is unused
def generate_random_v_change(v):
	stochastic = randint(-1, 1)/100
	velocities = np.linspace(-0.1, 0.1, 100)
	weighted_probs = np.exp(-1*(weighted_probs - v)**2) + stochastic
	weighted_probs_list = [val for val in weighted_probs]
	return choice(velocities, weights=weighted_probs)

# Updates the current velocity and position. This function is unused
def update_random_path(x, y, v_x, v_y):
	rand_v_x_change = generate_random_v_change(v_x)
	rand_v_y_change = generate_random_v_change(v_y)
	v_x += rand_v_x_change
	v_y += rand_v_y_change
	x += v_x*dt
	y += v_y*dt
	return x, y, v_x, v_y

# Apply a gaussian convolution to the STM
def create_mov_avg_from_stm(stm, curr_stm_size):
	mov_avg = np.zeros(SIDE_LEN, SIDE_LEN)
	memory_slots = np.arange(0, curr_stm_size)
	weights = np.exp(-1*(memory_slots - int(curr_stm_size/2))**2)
	for slot in range(0, curr_stm_size):
		mov_avg += weights[slot]*stm[i]
	return mov_avg

# Get a prediction score by comparing current STM and current BVC activity
def create_pred_score(stm_map, bvc_array):
	stm_map /= np.max(stm_map)
	bvc_array /= np.max(bvc_array)
	return np.dot(stm_map, bvc_array)

# Adds a new memory to the STM, or replaces an old one
def add_memory_to_stm(stm, curr_stm_size):
	if curr_stm_size < STM_RANGE:
		stm[curr_stm_size] = new_map
		curr_stm_size += 1
	else:
		stm = roll_back_stm()
		stm[-1] = new_map
	return stm, curr_stm_size

# Adds a new memory to the LTM if space exists
def add_memory_to_ltm(ltm, curr_ltm_size):
	if curr_ltm_size < MAX_MEMORIES:
		ltm[curr_ltm_size] = new_map
		curr_ltm_size += 1
	else:
		stm[randint(0, MAX_MEMORIES)] = new_map
	return ltm, curr_ltm_size

# Evaluates the surprisal level given the change from the previous position to the new one
def make_judgement_on_location(x, y, environment, stm, ltm, curr_stm_size, curr_ltm_size):
	bvc_array = update_bvc(x, y, environment)
	stm_avg = create_mov_avg_from_stm(stm, curr_stm_size)
	pred_score = create_pred_score(stm_avg, bvc_array)
	if pred_score > PRED_THRESHHOLD:
		stm = add_memory_to_stm(bvc_array, curr_stm_size)
	else:
		ltm, curr_ltm_size, was_successful = check_if_map_matches_any_current_entries()
		if was_successful:
			print("I've seen this before")
		else:
			ltm, curr_ltm_size = add_memory_to_ltm(ltm, curr_ltm_size)
			print("New room!")
	print("Current surprisal score: " + str(pred_score))
	return environment, stm, ltm, curr_stm_size, curr_ltm_size

# Initializes all variables, runs online navigation guided by user in pygame
def run_navigation(random_or_guided="guided"):
	# Get map of environment to traverse over visually
	map_of_environment = retriever.retrieve_map_of_environment()
	# Initialize the current sizes of the LTM and STM
	curr_ltm_size = 0
	curr_stm_size = 0
	# Initialize LTM and STM
	stm = np.zeros((STM_RANGE, TICKS))
	ltm = np.zeros((MAX_MEMORIES, TICKS))
	# Run program in pygame
	if random_or_guided="guided":
		gui.explore_environment(map_of_environment, stm, ltm, curr_stm_size, curr_ltm_size)
	else:
		print("Unable to randomly generate path.")
