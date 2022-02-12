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
	min_rad, max_rad = int(np.min(sensory_map)), int(np.max(sensory_map))
	# Range cannot handle floats so we iterate by whole numbers and divide
	for bvc_r in range(min_rad, max_rad, BVC_RAD_STEP):
		for bvc_theta_index in range(0, TICKS, BVC_TICK_STEP):
			bvc_theta = THETAS[bvc_theta_index]
			bvc_x = int(x + bvc_r*X_COMP[bvc_theta_index])
			bvc_y = int(y - bvc_r*Y_COMP[bvc_theta_index])
			bvc_cartesian_array[bvc_x][bvc_y] = get_total_freq(sensory_map, bvc_r, bvc_theta)
	return bvc_cartesian_array

# Integrate all values of r and theta across each BVC
def get_total_freq(sensory_map, preferred_rad, preferred_theta):
	total_freq = 0
	for tick in range(TICKS):
		theta = THETAS[tick]
		rad = sensory_map[tick]
		total_freq += freq_at_r_theta(rad, theta, preferred_rad, preferred_theta, sigma_angle=1)
	return total_freq

# Get the frequency contribution to a BVC at each distinct r and theta
def freq_at_r_theta(r, theta, d, phi, sigma_angle):
	sigma_rad = d
	r_component = np.exp((-1*(r-d)**2)/(2*sigma_rad**2))/np.sqrt(2*np.pi*sigma_rad**2)
	theta_component = np.exp((-1*(theta-phi)**2)/(2*sigma_angle**2))/np.sqrt(2*np.pi*sigma_angle**2)
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
	mov_avg = np.zeros((SIDE_LEN*SIDE_LEN))
	memory_slots = np.arange(0, curr_stm_size)
	weights = np.exp(-1*(memory_slots - int(curr_stm_size/2))**2)
	for slot in range(0, curr_stm_size):
		mov_avg += weights[slot]*stm[slot]
	#print("SUMMED AVG: " + str(np.sum(mov_avg)))
	return mov_avg

# Get a prediction score by comparing current STM and current BVC activity
def create_pred_score(stm_map, bvc_array):
	stm_map = stm_map.flatten()
	bvc_array = bvc_array.flatten()
	if np.max(stm_map) != 0:
		stm_map /= np.linalg.norm(stm_map)
	if np.max(bvc_array) != 0:
		bvc_array /= np.linalg.norm(bvc_array)
	return np.dot(stm_map, bvc_array.T)

# Adds a new memory to the STM, or replaces an old one
def add_memory_to_stm(bvc_array, stm, curr_stm_size):
	if curr_stm_size < STM_RANGE:
		stm[curr_stm_size] = bvc_array.flatten()
		curr_stm_size += 1
	else:
		stm = roll_back_stm()
		stm[-1] = bvc_array.flatten()
	return stm, curr_stm_size

# Adds a new memory to the LTM if space exists
def add_memory_to_ltm(bvc_array, ltm, curr_ltm_size):
	if curr_ltm_size < MAX_MEMORIES:
		ltm[curr_ltm_size] = bvc_array.flatten()
		curr_ltm_size += 1
	else:
		stm[randint(0, MAX_MEMORIES)] = bvc_array.flatten()
	return ltm, curr_ltm_size

# Currently, this function does not work. Once completed, it will evulate the current observation to previous ones in the LTM and see if any resemble it
def check_if_map_matches_any_current_entries(bvc_array, ltm):
	return False

# Evaluates the surprisal level given the change from the previous position to the new one
def make_judgement_on_location(x, y, environment, stm, ltm, curr_stm_size, curr_ltm_size):
	bvc_array = update_bvc(x, y, environment)
	stm_avg = create_mov_avg_from_stm(stm, curr_stm_size)
	pred_score = create_pred_score(stm_avg, bvc_array)
	stm, curr_stm_size = add_memory_to_stm(bvc_array, stm, curr_stm_size)
	if pred_score < PRED_THRESHHOLD:
		was_successful = check_if_map_matches_any_current_entries(bvc_array, ltm)
		if was_successful:
			print("I've seen this before")
		else:
			ltm, curr_ltm_size = add_memory_to_ltm(bvc_array, ltm, curr_ltm_size)
			print("New room!")
	print("Current surprisal score: " + str(pred_score))
	return environment, stm, ltm, curr_stm_size, curr_ltm_size

# Initializes all variables, runs online navigation guided by user in pygame
def run_navigation(random_or_guided="guided"):
	# Get map of environment to traverse over visually
	environment, x, y = retriever.retrieve_environment()
	# Initialize the current sizes of the LTM and STM
	curr_ltm_size = 0
	curr_stm_size = 0
	# Initialize LTM and STM
	stm = np.zeros((STM_RANGE, SIDE_LEN**2))
	ltm = np.zeros((MAX_MEMORIES, SIDE_LEN**2))
	# Run program in pygame
	if random_or_guided == "guided":
		gui.explore_environment(environment, stm, ltm, curr_stm_size, curr_ltm_size)
	else:
		print("Unable to randomly generate path.")

