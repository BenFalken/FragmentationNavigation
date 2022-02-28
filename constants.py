import numpy as np

# The pixel dimensions for the window in which your environment can be drawn
SIDE_LEN = 500
# For efficiency's sake, we choose ever tenth point (row, col) to construct a sensory map
STEP = 5

# The fixed euclidean distance at which surprisal is collected relative to the reference point
RADIUS = 2
# Because we are making a circle out of pixels, we add an error to ensure a ring of points is collected to compare
ERROR = 1

# Incremental value of tracker variable used to create spatial map
dt = 0.5

# Number of ticks in the spatial map generated
TICKS = 512

# Generated list of directions
THETAS = np.linspace(-1*np.pi, np.pi, TICKS)

# Multiply the number of ticks outward from a point by the appropriate element here, and you get the cartesian conversion
X_COMP = np.cos(THETAS)
Y_COMP = np.sin(THETAS)

# Paintbrush size for drawing environment
CURSOR_RAD = 5

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)

#DBSCAN param
EPS = 5

# Cutoff for what we define as a room, in terms of how many points are included. Clusters of less than this point number are ignored
MIN_POINTS_FOR_CLUSTER_TO_BE_CONSIDERED_REGION = 20

# Constants for STM, LTM
STM_RANGE = 10
MAX_MEMORIES = 10

# Constants for physical environment grid cell conversion
NUM_GRID_CELLS = 1
BVC_PER_ROW, BVC_PER_COL = 100, 100
PIXEL_PER_BVC = int(SIDE_LEN/BVC_PER_ROW)

ENVIRONMENT_DIMEN = 5 	# meters
METERS_TO_PIXELS = int(SIDE_LEN/ENVIRONMENT_DIMEN)

# Decided based on my own measurements. 0.9 is a li'l high
PRED_THRESHHOLD = 0.5

# How to compare the current observation and a memory in ltm, to see if you've been in the same room before
CERTAINTY_SCORE_THAT_MEMORIES_MATCH = 0.8

# Some constants to describe how far and how often BVCs exist in terms of distance from the subject
BVC_RADIUS = int(SIDE_LEN/2)
BVC_RAD_STEP = 2
BVC_TICK_STEP = 4

# If a user hasn't clicked to add a new point to the path in online navigation, we just use this as a stand-in
INVALID_COORD = -1000

# Parameters for the grid cells
NUM_MODULES = 4
LAMBDA_0 = ENVIRONMENT_DIMEN
PERIOD_RATIO = 0.5

MAX_SPIKE_COUNT = 1

# Centers

CENTERS_PER_ROW = 5
CENTERS_PER_COL = 5

PIXELS_PER_CENTER = int(SIDE_LEN/CENTERS_PER_ROW)
