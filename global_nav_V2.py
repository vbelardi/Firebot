import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap

#========================================= Constants =========================================
DEBUG_ENABLED = False
PIXEL_X = 480
PIXEL_Y = 640
REAL_X_CM = 72
REAL_Y_CM = 107
BOX_SIZE_X = 72
BOX_SIZE_Y = 107

#========================================= Test data =========================================

# black_rectangle = [(336, 331, 471, 491), (523, 311, 660, 481), (123, 287, 255, 483), (99, 172, 289, 296), (338, -11, 478, 237), (121, -11, 309, 119)]
# blue_rectangle = (68, 392)
# green_rectangle = [(568, 76, 603, 111)]
# red_rectangle = [(134, 241, 237, 344), (337, 179, 471, 289), (424, 119, 527, 222), (-6, -6, 120, 124)]

#========================================= functions =========================================


def process_rectangles(shapes, rectangles, shape_key, pixel_x=PIXEL_X, pixel_y=PIXEL_Y):
    """
    Process the rectangles and add them to the shapes dictionary.

    Parameters:
    shapes (dict): The dictionary to store the shapes.
    rectangles (list): The list of rectangles.
    shape_key (str): The key to store the shape in the shapes dictionary.
    pixel_x (int): The numbers of pixel in the x-coordinate.
    pixel_y (int): The number of pixel in the y-coordinate.
    """
    for rect in rectangles:
        start_0 = int((rect[1] / pixel_x) * BOX_SIZE_X)
        start_1 = int((rect[0] / pixel_y) * BOX_SIZE_Y)
        end_0 = int((rect[3] / pixel_x) * BOX_SIZE_X)
        end_1 = int((rect[2] / pixel_y) * BOX_SIZE_Y)
        if start_0 < 0:
            start_0 = 0
        if start_1 < 0:
            start_1 = 0
        if end_0 > BOX_SIZE_X-1:
            end_0 = BOX_SIZE_X-1
        if end_1 > BOX_SIZE_Y-1:
            end_1 = BOX_SIZE_Y-1
        
        if shape_key == "goal":
            new_start = int((start_0 + end_0) / 2)
            new_end = int((start_1 + end_1) / 2)
            shapes[shape_key].append(((new_start, new_end), (new_start, new_end)))
        else:
            shapes[shape_key].append(((start_0, start_1), (end_0, end_1)))


def place_shape(grid, top_left, bottom_right, color_code):
    """
    Place the shape in the grid.

    Parameters:
    grid (numpy.ndarray): The grid with character values indicating different areas.
    top_left (tuple): The top left corner of the shape.
    bottom_right (tuple): The bottom right corner of the shape.
    color_code (str): The color code of the shape.
    """
    for i in range(top_left[0], bottom_right[0] + 1):
        for j in range(top_left[1], bottom_right[1] + 1):
            grid[i, j] = color_code  # Note: grid[y, x] because numpy is row-major


def determine_board(shapes, black_rectangle, blue_rectangle, green_rectangle, red_rectangle, pixel_x, pixel_y):
    """
    Determine the board based on the rectangles.

    Parameters:
    shapes (dict): The dictionary to store the shapes.
    black_rectangle (list): The list of black obstacles.
    blue_rectangle (tuple): The starting point of the thymio.
    green_rectangle (list): The goal.
    red_rectangle (list): The list of red obstacles.
    pixel_x (int): The numbers of pixel in the x-coordinate.
    pixel_y (int): The number of pixel in the y-coordinate.
    """
    process_rectangles(shapes, black_rectangle, "wall", pixel_x, pixel_y)

    blue_rectangle_0 = int((blue_rectangle[1] / pixel_x) * BOX_SIZE_X)
    blue_rectangle_1 = int((blue_rectangle[0] / pixel_y) * BOX_SIZE_Y)
    shapes["start"].append(((blue_rectangle_0, blue_rectangle_1), (blue_rectangle_0, blue_rectangle_1)))

    process_rectangles(shapes, green_rectangle, "goal", pixel_x, pixel_y)
    process_rectangles(shapes, red_rectangle, "fire", pixel_x, pixel_y)


def mark_hazardous_areas(grid_char):
    """
    Mark the hazardous areas in the grid with 'h' and 'd' based on the fire and the distance from the fire.

    Parameters:
    grid_char (numpy.ndarray): The grid with character values indicating different areas.
    """
    for y in range(grid_char.shape[0]):
        for x in range(grid_char.shape[1]):
            if grid_char[y, x] != 'f': 
                continue
            for i in range(-3, 4):
                for j in range(-3, 4):
                    if not (i or j): continue
                    if 0 <= x + i < grid_char.shape[1] and 0 <= y + j < grid_char.shape[0] and grid_char[y + j, x + i] == 'n':

                        if grid_char[y + j, x + i] == 'g' or grid_char[y + j, x + i] == 's':
                            continue

                        #mark cells around fire as hazardous
                        if abs(i) <= 2 and abs(j) <= 2 :
                            grid_char[y + j, x + i] = 'h'

                        #mark cells around hazardous cells as dangerous
                        else :
                            grid_char[y + j, x + i] = 'd'


def increment_value(grid, grid_char, x, y, value):
    """
    Increment the value in the grid based on the surrounding cells and the grid_char.

    Parameters:
    grid (numpy.ndarray): The grid with numerical values.
    grid_char (numpy.ndarray): The grid with character values indicating different areas.
    x (int): The x-coordinate of the current cell.
    y (int): The y-coordinate of the current cell.
    value (int): The value to be incremented and propagated.
    """
    queue = [(x, y, value)]
    while queue:
        x, y, value = queue.pop(0)
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0: continue
                # don't allow diagonal increments
                if abs(i) == 1 and abs(j) == 1: continue

                if 0 <= x + i < grid.shape[1] and 0 <= y + j < grid.shape[0]:

                    #return if goal is reached
                    if grid_char[y + j, x + i] == 'g':
                        grid[y + j, x + i] = value
                        return
                    
                    if (value < grid[y + j, x + i] or grid[y + j, x + i] == 0):
                        # increment value by 1 if the cell is not a wall
                        if grid_char[y + j, x + i] == 'n':
                            grid[y + j, x + i] = value

                        # increment value by 2 if the cell is dangerous
                        elif grid_char[y + j, x + i] == 'd':
                            value += 2
                            grid[y + j, x + i] = value

                        # increment value by 2 if the cell is hazardous
                        elif grid_char[y + j, x + i] == 'h':
                            value += 4
                            grid[y + j, x + i] = value

                        else:
                            continue

                        queue.append((x + i, y + j, value + 1))
                       

def populate_grid(grid_num, grid_char):
    """
    Fill the grid with numerical values based on the grid_char.
    
    Parameters:
    grid_num (numpy.ndarray): The grid with numerical values.
    grid_char (numpy.ndarray): The grid with character values indicating different areas.
    """
    x, y = -1, -1
    for i in range(grid_num.shape[0]):
        for j in range(grid_num.shape[1]):
            if grid_char[i, j] == 's':
                x, y = j, i
                # print("Start found", x, y)
                increment_value(grid_num, grid_char, x, y, 1)
                return


def step_backwards(grid_num, grid_char, x, y):
    """
    Step backwards in the grid to find the path from the goal to the start.

    Parameters:
    grid_num (numpy.ndarray): The grid with numerical values.
    grid_char (numpy.ndarray): The grid with character values indicating different areas.
    x (int): The x-coordinate of the current cell.
    y (int): The y-coordinate of the current cell.
    """
    new_x, new_y = x, y
    if grid_char[y, x] == 'g':
        new_value = 1000
    else:
        new_value = grid_num[y, x]

    for i in range(-1, 2):
        for j in range(-1, 2):
            if not (i or j): continue

            if 0 <= x + i < grid_num.shape[1] and 0 <= y + j < grid_num.shape[0]:
                # return if the start is reached
                if grid_char[y + j, x + i] == 's':
                    return x + i, y + j
                
                # find the cell with the smallest value
                if grid_num[y + j, x + i] < new_value and grid_num[y + j, x + i] != 0:
                    new_x, new_y = x + i, y + j
                    new_value = grid_num[y + j, x + i]

    return new_x, new_y
            

def find_path(grid_num, grid_char, path):
    """
    Find the path from the goal to the start by stepping backwards through the grid.

    grid_num (numpy.ndarray): The grid with numerical values.
    grid_char (numpy.ndarray): The grid with character values indicating different areas.
    path (list): The list to store the path coordinates.
    """
    x, y = -1, -1
    for i in range(grid_num.shape[1]):
        for j in range(grid_num.shape[0]):
            if grid_char[j, i] == 'g':
                x, y = i, j
                break
        if x != -1 and y != -1:
            break

    while grid_char[y, x] != 's':
        x, y = step_backwards(grid_num, grid_char, x, y)
        path.append([x, y])


def find_change_directions(path):
    """
    Find the change in direction in the path.

    Parameters:
    path (list): The list of coordinates in the path.

    Returns:
    change_directions: The list of change in directions.
    """
    change_directions = []

    current_direction = path[0]
    change_directions.append(current_direction)

    for i, coor in enumerate(path):
        if coor[0] == current_direction[0] or coor[1] == current_direction[1]:
            continue
        elif abs(round(coor[0] - current_direction[0])) ==  abs(round(coor[1] - current_direction[1])):
            continue
        else:
            current_direction = path[i-1]
            change_directions.append(current_direction)

    change_directions.append(path[-1])

    return change_directions


def convert_to_real_coordinates(path):
    """
    Convert the grid coordinates to real-world coordinates.

    Parameters:
    path (int, int): The x-coordinate and y-coordinates of the grid.

    Returns:
    path_cm: The real-world coordinates.
    """
    inverted_path = np.zeros_like(path)
    path_cm = np.zeros_like(path, dtype=float)

    for i in range(len(path)):
        inverted_path[i][0] = path[len(path)-1-i][0]
        inverted_path[i][1] = path[len(path)-1-i][1]
    
    for i, coor in enumerate(inverted_path):
        path_cm[i][0] = (float(coor[0] / BOX_SIZE_Y)) * REAL_Y_CM
        path_cm[i][1] = (float(coor[1] / BOX_SIZE_X)) * REAL_X_CM
    return path_cm

#========================================= Desiplay path to debug =========================================

def display_grid(grid, grid_char, path, new_path):
    """
    Display the grid with the path and the new path.

    Parameters:
    grid (numpy.ndarray): The grid with numerical values.
    grid_char (numpy.ndarray): The grid with character values indicating different areas.
    path (list): The list of coordinates in the path.
    new_path (list): The list of coordinates in the new path.
    """
    colors = ['white']
    cmap = ListedColormap(colors)
    fig, ax = plt.subplots()
    ax.imshow(np.zeros_like(grid), cmap=cmap)  # Display a blank grid

    # Annotate each cell with the corresponding color
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid_char[i, j] == 's':
                rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1, linewidth=1, edgecolor='none', facecolor='blue', alpha=1)
                ax.add_patch(rect)
            elif grid_char[i, j] == 'g':
                rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1, linewidth=1, edgecolor='none', facecolor='green', alpha=1)
                ax.add_patch(rect)
            elif grid_char[i, j] == 'f':
                rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1, linewidth=1, edgecolor='none', facecolor='red', alpha=1)
                ax.add_patch(rect)
            elif grid_char[i, j] == 'w':
                rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1, linewidth=1, edgecolor='none', facecolor='black', alpha=1)
                ax.add_patch(rect)
            elif grid_char[i, j] == 'h' or grid_char[i, j] == 'd':
                rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1, linewidth=1, edgecolor='none', facecolor='orange', alpha=0.5)
                ax.add_patch(rect)
    
    # Display the path and the new path
    for coord in path:
        if len(coord) == 0: continue
        y, x = coord
        rect = patches.Rectangle((y - 0.5, x - 0.5), 1, 1, linewidth=1, edgecolor='none', facecolor='lightblue', alpha=0.5)
        ax.add_patch(rect)

    for coord in new_path:
        if len(coord) == 0: continue
        y, x = coord
        rect = patches.Rectangle((y - 0.5, x - 0.5), 1, 1, linewidth=1, edgecolor='none', facecolor='blue', alpha=1)
        ax.add_patch(rect)

    plt.show()

#========================================= Main function =========================================

def global_navigation_algorithm(black_rectangle, blue_rectangle, green_rectangle, red_rectangle, pixel_x, pixel_y):
    """
    The main function to run the global navigation algorithm.

    Parameters:
    black_rectangle (list): The list of black obstacles.
    blue_rectangle (tuple): The starting point of the thymio.
    green_rectangle (list): The goal.
    red_rectangle (list): The list of red obstacles.
    pixel_x (int): The numbers of pixel in the x-coordinate.
    pixel_y (int): The number of pixel in the y-coordinate.

    Returns:
    reduced_path: The reduced path with only the change in directions.
    """

    grid_num = np.zeros((BOX_SIZE_X, BOX_SIZE_Y))
    grid_char = np.full((BOX_SIZE_X, BOX_SIZE_Y), 'n')
    path = []

    shapes = {
        "wall": [],
        "start": [],
        "goal": [],
        "fire": []
    }

    # Define the color codes for each shape
    item_character_codes = {
        "wall": 'w',
        "start": 's',
        "goal": 'g',
        "fire": 'f',
    }

    determine_board(shapes, black_rectangle, blue_rectangle, green_rectangle, red_rectangle, pixel_x, pixel_y)

    # Place each shape in the grid
    for color, coordinates in shapes.items():
        for top_left, bottom_right in coordinates:
            place_shape(grid_char, top_left, bottom_right, item_character_codes[color])

    # display the computing time for each function in debug mode
    if DEBUG_ENABLED:
        start_time = time.time()
        mark_hazardous_areas(grid_char)
        end_time = time.time()

        execution_time = end_time - start_time
        print(f"mark_hazardous_areas function executed in {execution_time:.4f} seconds.")

        # display_grid_with_numbers(grid_num, grid_char, path, path)

        print("Fill grid with numbers")
        start_time = time.time()
        populate_grid(grid_num, grid_char)
        end_time = time.time()

        execution_time = end_time - start_time
        print(f"fill_up function executed in {execution_time:.4f} seconds.")

        # disp(grid_num)

        print("Find path")
        start_time = time.time()
        find_path(grid_num, grid_char, path)
        end_time = time.time()

        execution_time = end_time - start_time
        print(f"find_path function executed in {execution_time:.4f} seconds.")
    
    else:
        mark_hazardous_areas(grid_char)
        populate_grid(grid_num, grid_char)
        find_path(grid_num, grid_char, path)

    print("Found path")

    path_cm = convert_to_real_coordinates(path)
    reduced_path = find_change_directions(path_cm)

    display_grid(grid_num, grid_char, path, reduced_path)

    return reduced_path

#========================================= Test the global navigation algorithm =========================================
# global_navigation_algorithm(black_rectangle, blue_rectangle, green_rectangle, red_rectangle, PIXEL_X, PIXEL_Y)
