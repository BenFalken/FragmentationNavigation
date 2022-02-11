import pygame
from constants import *

# Using a cursor-driven framework, create a map of an envionment in the form of a one-and-zero array
def draw_map():
    pygame.init()
    screen = pygame.display.set_mode((SIDE_LEN, SIDE_LEN))
    screen.fill(WHITE)
    running, pressed = True, False
    environment = np.ones((SIDE_LEN, SIDE_LEN))
    # Continouly check if the user is dragging/clicking the cursor. If so, update the screen and data
    play_message()
    while running:
        for event in pygame.event.get():
           environment, x, y = handle_key_press(event, environment)
        pygame.display.flip()
    pygame.quit()
    return environment, x, y

# Tell the user about the program
def play_message():
    print("Please use your cursor to draw an appropriate map to navigate!")
    print("Drag your mouse while pressing it to 'dig' through white areas. White areas act as boundaries/walls.")
    print("Press any key once finished.")
    print("*************************")

def play_online_message():
    print("You will now navigate the environment. Click in any black space to move, constructing a path as you go.")

# Handle a key press event by declaring the state of pressed, and update screen with new values as required
def handle_key_press(event, environment):
    if event.type == pygame.QUIT:
        running = False
    if event.type == pygame.KEYDOWN:
        running = False
        [x, y] = pygame.mouse.get_pos()
        return environment, x, y
    if event.type == pygame.MOUSEBUTTONDOWN:
        pressed = True
    if event.type == pygame.MOUSEBUTTONUP:
        pressed = False
    if event.type == pygame.MOUSEMOTION and pressed:
        environment = update_screen(environment)

# Draws out the environment after having stored it
def draw_environment(environment):
    for row in range(environment.shape[0]):
        for col in range(environment.shape[1]):
            if environment[row][col] == 0:
                pygame.draw_rect(screen, BLACK, (col, row, 1, 1))

# If a key was pressed, add the position to the online path
def handle_online_key_press(event, environment, all_points):
    if event.type == pygame.QUIT:
        running = False
    if event.type == pygame.KEYDOWN:
        [x, y] = pygame.mouse.get_pos()
        if environment[y][x] == 0:
            update_online_screen(x, y)
            all_points.append([x, y])
            return x, y, all_points
        else:
            return INVALID_COORD, INVALID_COORD, all_points

def update_online_screen(x, y):
    pygame.draw.line(screen, GREEN, all_points[-1], [x, y])
    pygame.draw.circle(screen, RED, [x, y], 2)
    return x, y

# Add values to the array of the environment and update the screen
def update_screen(environment):
    [x, y] = pygame.mouse.get_pos()
    for i in range(-1*CURSOR_RAD, CURSOR_RAD + 1):
        for j in range(-1*CURSOR_RAD, CURSOR_RAD + 1):
            environment[clamp(y+i, SIDE_LEN - 1)][clamp(x+j, SIDE_LEN - 1)] = 0
            pygame.draw.rect(screen, BLACK, (x-CURSOR_RAD, y-CURSOR_RAD, CURSOR_RAD*2, CURSOR_RAD*2))
    return environment

def explore_environment(environment, stm, ltm, curr_stm_size, curr_ltm_size):
    all_points = []
    pygame.init()
    screen = pygame.display.set_mode((SIDE_LEN, SIDE_LEN))
    screen.fill(WHITE)
    running, pressed = True, False
    play_online_message()
    while running:
        for event in pygame.event.get():
           x, y, all_points = handle_online_key_press(event, environment, all_points)
           if (x, y) != (INVALID_COORD, INVALID_COORD):
            environment, stm, ltm, curr_stm_size, curr_ltm_size = offline_navigation.make_judgement_on_location(x, y, environment, stm, ltm, curr_stm_size, curr_ltm_size)
        pygame.display.flip()
    pygame.quit()