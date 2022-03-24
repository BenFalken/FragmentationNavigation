import gui, retriever
import numpy as np
import matplotlib.pyplot as plt
from random import randint, choice
from constants import *
from utilities import *
from grid_cell import GridCell


# Just tells the user to stand by
def print_grid_cell_message():
	print("************************************************************")
	print("*")
	print("*")
	print("*")
	print("WE WILL NOW CREATE THE GRID CELLS FOR THE ONLINE NAVIGATION.")
	print("PLEASE WAIT.")
	print("*")
	print("*")
	print("*")
	print("************************************************************")

# Announce a new grid cell was created
def announce_new_grid_cell(index):
	print("*")
	print("Grid Cell " + str(index+1) + " created")
	print("*")
	print("*******************")

# Self explanatory. Makes the grid cells
def create_all_grid_cells():
	print_grid_cell_message()
	grid_cells = []
	for i in range(NUM_GRID_CELLS):
		new_cell = GridCell(lambda_m=0.125*(i+1))
		new_cell.construct_firing_map()
		grid_cells.append(new_cell)
		announce_new_grid_cell(i)
	return grid_cells

# Graphs the firing maps
def show_all_firing_maps(grid_cells):
	for cell in grid_cells:
		plt.title("Grid cell firing map for the current grid cell")
		plt.imshow(cell.firing_map)
		plt.show()

# To be implemented later on once I figure out how to represent grid cell modules as SDRs
def get_binary_representation_of_location(grid_cells):
	print("Not now sir")

# Make a cartesian array of BVCs and find the frequencies of each one given the current bounds in terms of r, theta
def update_bvc(x, y, environment):
	bvc_cartesian_array = np.zeros((BVC_PER_ROW, BVC_PER_COL))
	sensory_map = generate_sensory_map(x, y, environment)
	min_rad, max_rad = 0, int(np.max(sensory_map)) #int(np.min(sensory_map)), int(np.max(sensory_map))
	# Range cannot handle floats so we iterate by whole numbers and divide
	for tick in range(TICKS):
		theta = THETAS[tick]
		r = sensory_map[tick]
		for tick_for_each_r in range(0, TICKS, BVC_TICK_STEP):
			theta_for_each_r = THETAS[tick_for_each_r]
			bvc_x = int(x + r*X_COMP[tick_for_each_r])
			bvc_x = int((bvc_x - bvc_x%PIXEL_PER_BVC)/PIXEL_PER_BVC)

			bvc_y = int(y - r*Y_COMP[tick_for_each_r])
			bvc_y = int((bvc_y - bvc_y%PIXEL_PER_BVC)/PIXEL_PER_BVC)
			try:
				bvc_cartesian_array[bvc_y][bvc_x] = get_total_freq(sensory_map, r, theta, theta_for_each_r)
			except:
				continue

	return bvc_cartesian_array

# Integrate all values of r and theta across each BVC
def get_total_freq(sensory_map, r, theta, theta_for_each_r):
	freq_theta = freq_at_theta(theta, theta_for_each_r, sigma_angle=0.005)
	freq_r = 1
	#freq_r = freq_at_r(r, r)
	return freq_theta*freq_r

# Get the frequency contribution to a BVC at each distinct theta
def freq_at_theta(theta, phi, sigma_angle):
	theta_component = np.exp((-1*(theta - phi)**2)/sigma_angle)
	return theta_component

# Get the frequency contribution to a BVC at each distinct r
def freq_at_r(r, d):
	sigma_rad = d
	r_component = np.exp(-1*sigma_rad*(d - r)**2)
	return r_component

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
	mov_avg = np.zeros((BVC_PER_ROW*BVC_PER_COL))
	memory_slots = np.arange(0, curr_stm_size)
	weights = np.exp(-1*(memory_slots - int(curr_stm_size/2))**2)
	for slot in range(0, curr_stm_size):
		mov_avg += weights[slot]*stm[slot]
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

# Roll back the stm to shift it
def roll_back_stm(stm):
	rolled_stm = np.zeros(stm.shape)
	for slot in range(1, STM_RANGE):
		rolled_stm[slot - 1] = stm[slot]
	rolled_stm[-1] = stm[0]
	return rolled_stm

# Adds a new memory to the STM, or replaces an old one
def add_memory_to_stm(bvc_array, stm, curr_stm_size):
	if curr_stm_size < STM_RANGE:
		stm[curr_stm_size] = bvc_array.flatten()
		curr_stm_size += 1
	else:
		stm = roll_back_stm(stm)
		stm[-1] = bvc_array.flatten()
	return stm, curr_stm_size

# Adds a new memory to the LTM if space exists
def add_memory_to_ltm(bvc_array, ltm, curr_ltm_size):
	if curr_ltm_size < MAX_MEMORIES:
		ltm[curr_ltm_size] = bvc_array.flatten()
		curr_ltm_size += 1
	else:
		ltm[randint(0, MAX_MEMORIES - 1)] = bvc_array.flatten()
	return ltm, curr_ltm_size

# Basically just takes the dot product of the current bvc values and all memories in the ltm, finding the closest approximation
def check_if_map_matches_any_current_entries(bvc_array, ltm):
	normalized_bvc_array = bvc_array.flatten() / np.linalg.norm(bvc_array.flatten())
	for memory in ltm:
		normalized_memory = memory
		if np.max(normalized_memory) != 0:
			normalized_memory = memory / np.linalg.norm(memory)
		if np.dot(normalized_memory, normalized_bvc_array.T) >= CERTAINTY_SCORE_THAT_MEMORIES_MATCH:
			return True
	return False

# Present the averaged BVC array STM
def present_stm_avg(stm_avg):
	plt.title("BVC memory at the current position")
	plt.imshow(stm_avg.reshape((BVC_PER_ROW, BVC_PER_COL)))
	plt.show()

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
	present_stm_avg(stm_avg)
	print("Current surprisal score: " + str(pred_score))
	return environment, stm, ltm, curr_stm_size, curr_ltm_size

# Initializes all variables, runs online navigation guided by user in pygame
def run_navigation(random_or_guided="guided"):
	grid_cells = retriever.retrieve_grid_cells()
	show_all_firing_maps(grid_cells)
	# Get map of environment to traverse over visually
	environment, x, y = retriever.retrieve_environment()
	# Initialize the current sizes of the LTM and STM
	curr_ltm_size = 0
	curr_stm_size = 0
	# Initialize LTM and STM
	stm = np.zeros((STM_RANGE, BVC_PER_ROW**2))
	ltm = np.zeros((MAX_MEMORIES, BVC_PER_ROW**2))
	# Run program in pygame
	if random_or_guided == "guided":
		gui.explore_environment(environment, stm, ltm, curr_stm_size, curr_ltm_size)
	else:
		print("Unable to randomly generate path.")

