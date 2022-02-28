import pygame
import online_navigation
from constants import *
from utilities import *

# Using a cursor-driven framework, create a map of an envionment in the form of a one-and-zero array
def draw_map():
    pygame.init()
    screen = pygame.display.set_mode((SIDE_LEN, SIDE_LEN))
    screen.fill(WHITE)
    running, pressed = True, False
    environment = np.ones((SIDE_LEN, SIDE_LEN))
    # Continously check if the user is dragging/clicking the cursor. If so, update the screen and data
    play_message()
    while running:
        for event in pygame.event.get():
           x, y, environment, running, pressed = handle_key_press(screen, event, environment, running, pressed)
        pygame.display.flip()
    pygame.quit()
    return environment, x, y

# Tell the user about the program
def play_message():
    print("***************************************************************")
    print("*")
    print("*")
    print("*")
    print("Please use your cursor to draw an appropriate map to navigate!")
    print("Drag your mouse while pressing it to 'dig' through white areas. White areas act as boundaries/walls.")
    print("Press any key once finished.")
    print("*")
    print("*")
    print("*")
    print("***************************************************************")

def play_online_message():
    print("***************************************************************")
    print("*")
    print("*")
    print("*")
    print("You will now navigate the environment. Click any key in any black space to move, constructing a path as you go.")
    print("*")
    print("*")
    print("*")
    print("***************************************************************")

# Handle a key press event by declaring the state of pressed, and update screen with new values as required
def handle_key_press(screen, event, environment, running, pressed):
    if event.type == pygame.QUIT:
        running = False
    if event.type == pygame.KEYDOWN:
        running = False
        [x, y] = pygame.mouse.get_pos()
        return x, y, environment, running, pressed
    if event.type == pygame.MOUSEBUTTONDOWN:
        pressed = True
    if event.type == pygame.MOUSEBUTTONUP:
        pressed = False
    if event.type == pygame.MOUSEMOTION and pressed:
        environment = update_screen(screen, environment)
    return INVALID_COORD, INVALID_COORD, environment, running, pressed

# Draws out the environment after having stored it
def draw_online_environment(screen, environment):
    for row in range(environment.shape[0]):
        for col in range(environment.shape[1]):
            if environment[row][col] == 0:
                pygame.draw.rect(screen, BLACK, (col, row, 1, 1))

# If a key was pressed, add the position to the online path. Otherwise, just return invalid data
def handle_online_key_press(screen, event, environment, all_points, running):
    if event.type == pygame.QUIT:
        running = False
    if event.type == pygame.KEYDOWN:
        [x, y] = pygame.mouse.get_pos()
        if environment[y][x] == 0:
            update_online_screen(screen, all_points, x, y)
            all_points.append([x, y])
            return x, y, all_points
        else:
            return INVALID_COORD, INVALID_COORD, all_points
    return INVALID_COORD, INVALID_COORD, all_points

# Add the points denoting the path
def update_online_screen(screen, all_points, x, y):
    if len(all_points) > 0:
        pygame.draw.line(screen, GREEN, all_points[-1], [x, y])
    pygame.draw.circle(screen, RED, [x, y], 2)
    return x, y

# Add values to the array of the environment and update the screen
def update_screen(screen, environment):
    [x, y] = pygame.mouse.get_pos()
    for i in range(-1*CURSOR_RAD, CURSOR_RAD + 1):
        for j in range(-1*CURSOR_RAD, CURSOR_RAD + 1):
            environment[clamp(y+i, SIDE_LEN - 1)][clamp(x+j, SIDE_LEN - 1)] = 0
            pygame.draw.rect(screen, BLACK, (x-CURSOR_RAD, y-CURSOR_RAD, CURSOR_RAD*2, CURSOR_RAD*2))
    return environment

# Main function. Every tick of the program, check to see if the user has added onto the path. If so, update the state of the program
def explore_environment(environment, stm, ltm, curr_stm_size, curr_ltm_size):
    all_points = []
    pygame.init()
    screen = pygame.display.set_mode((SIDE_LEN, SIDE_LEN))
    screen.fill(WHITE)
    running, pressed = True, False
    play_online_message()
    draw_online_environment(screen, environment)
    while running:
        for event in pygame.event.get():
           x, y, all_points = handle_online_key_press(screen, event, environment, all_points, running)
           if (x, y) != (INVALID_COORD, INVALID_COORD):
            environment, stm, ltm, curr_stm_size, curr_ltm_size = online_navigation.make_judgement_on_location(x, y, environment, stm, ltm, curr_stm_size, curr_ltm_size)
        pygame.display.flip()
    pygame.quit()