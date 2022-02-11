import numpy as np

# Constants for environment boundaries, resolution
SIDE_LEN = 500
STEP = 5

# Parameters for calculating surprisal
RADIUS = 10
ERROR = 1

# Incremental value of tracker variable used to create spatial map
dt = 0.5

# Number of ticks in the spatial map generated
TICKS = 512

# Generated list of directions
THETAS = np.linspace(-1*np.pi, np.pi, TICKS)

# Step components that each reference point uses to step away towards the edges of the map
X_COMP = np.cos(THETAS)
Y_COMP = np.sin(THETAS)

# Drawer size
CURSOR_RAD = 5

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

#DBSCAN param
EPS = 6

# Constants for STM, LTM
STM_RANGE = 10
MAX_MEMORIES = 10

# Constants for physical environment
PIXEL_TO_METER = 25 # pixels to meters
ENVIRONMENT_DIMEN = 5 # meters
PIXEL_PER_GRID_CELL = 5

GRID_CELL_PER_ROW = int(SIDE_LEN/PIXEL_PER_GRID_CELL)
GRID_CELL_PER_COL = int(SIDE_LEN/PIXEL_PER_GRID_CELL)

PRED_THRESHHOLD = 0.9

BVC_RADIUS = int(SIDE_LEN/2)

INVALID_COORD = -1000