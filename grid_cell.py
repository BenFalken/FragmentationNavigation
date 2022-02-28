from constants import*
from utilities import *

class GridCell:
	def __init__(self, lambda_m=0.5, theta=0):
		self.lambda_m = lambda_m*METERS_TO_PIXELS
		self.theta = theta
		self.firing_map = np.zeros((int(SIDE_LEN), int(SIDE_LEN)))
		self.rad = 50
		self.diam = 2*self.rad
		self.init_centers()

	# Optional smoothing. I don't use this
	def smoothing(self, val):
		return np.exp(0.3*(val+1.5))

	# Sum the three consinusoidal waves to get the hexagonal pattern, scaled. thetas not yet implemented
	def get_magnitude_of_firing_map_at_point(self, x, y, center):
		center_x, center_y = center[0], center[1]
		displacement_mag = np.sqrt(((x - (center_x)) **2 + (y - (center_y))**2))
		self.firing_map[clamp(y, self.firing_map.shape[0] - 1)][clamp(x, self.firing_map.shape[1] - 1)] += np.exp(-1*(displacement_mag**2)/self.lambda_m)
		theta_base = np.pi/3
		for k in range(-2, 1):
			theta_k = theta_base*k
			offset_x = int(np.cos(theta_k)*self.rad)
			offset_y = int(np.sin(theta_k)*self.rad)
			displacement_mag = np.sqrt(((x - (center_x + offset_x)) **2 + (y - (center_y + offset_y))**2))
			self.firing_map[clamp(y + offset_y, self.firing_map.shape[1] - 1)][clamp(x + offset_x, self.firing_map.shape[0] - 1)] += np.exp(-1*(displacement_mag**2)/self.lambda_m)

	# At a particular center, calulate the value of the grid cell firing map
	def get_firing_map_at_center(self, center):
		for y in range(0, SIDE_LEN):
			for x in range(0, SIDE_LEN):
				self.get_magnitude_of_firing_map_at_point(x, y, center)

	# Create the equally spaced centers of the grid cells
	def init_centers(self):
		self.centers = []
		y_step = int(2*self.diam*np.sin(np.pi/3))
		x_step = int(2*self.diam)
		for y in range(0, SIDE_LEN + y_step, y_step):
			for x in range(0, SIDE_LEN, x_step):
				self.centers.append([x, y])

	# For each center in the grid cell's centers, add the center to the firing map
	def construct_firing_map(self):
		for center in self.centers:
			self.get_firing_map_at_center(center)
		