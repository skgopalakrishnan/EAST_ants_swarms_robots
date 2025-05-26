from microbit.ACO_test import get_viable_moves, select_move, step, \
    reverse_last_move, remove_backtracks, drop_pheromones_single, drop_pheromone
    
from microbit.ACO_test import WalledMaze

# on_start/Initialisation
moves_taken = []
width = 5
height = 5
start = [0, 0]
goal = [4, 4]
init_pher = 0.001  # Initial pheromone level - prevent division by zero
deposit_amount = 1  # Amount of pheromone to deposit
epsilon = 0.1
pos = start
path = [start]
scan_bools = [False, False, False, False]
allowed_moves = [0, 90, 180, 270]  # Representing moves as angles
orientation = 0  # 0: East, 90: North, 180: West, 270: South
num_reverses = 0  # Counter for reverses
# basic.delay(2000)
# basic.show_leds("""
#     # # # # #
#     # . . . .
#     # # # # .
#     # . . . .
#     # # # # #
#     """)

ph_array = [[init_pher for _ in range(width)] for _ in range(height)]  # 2D pheromone array

# basic.show_arrow(ArrowNames.NORTH)

#####
# Scan the environment and update scan_bools based on the grayscale sensor state
# def scan_environment():
#     scan_bools = [False, False, False, False]  # Reset scan_bools
#     CutebotPro.trackbit_state_value()
#     if CutebotPro.get_grayscale_sensor_state(TrackbitStateType.TRACKING_STATE_1):
#         # Straight path
#         scan_bools[0] = True
#     elif CutebotPro.get_grayscale_sensor_state(TrackbitStateType.TRACKING_STATE_9):
#         # Left turn
#         scan_bools[2] = True
#     elif CutebotPro.get_grayscale_sensor_state(TrackbitStateType.TRACKING_STATE_13):
#         # Right turn
#         scan_bools[3] = True
#     elif CutebotPro.get_grayscale_sensor_state(TrackbitStateType.TRACKING_STATE_0):
#         # Reverse
#         scan_bools[1] = True
#     else:
#         # No path   
#         pass
#     return scan_bools

#####

def scan_environment_test(pos, maze, orientation):
    
    scan_bools = [False, False, False, False] # Reset scan_bools
    
    maze_row, maze_col = WalledMaze.to_maze_coords(pos, maze.size)  # Convert to maze coordinates
    filtered_moves = []
    for move in allowed_moves:  # Iterate over allowed moves
        new_pos = step(move, orientation, [maze_row, maze_col])
        r, c = new_pos  # Get the new position coordinates
        if maze.grid[r][c] == 0:  # Check if the state is obstruction-free
            filtered_moves.append(move)
    
    for move in filtered_moves:
        if move == 0: 
            scan_bools[0] = True  # Forward
        elif move == 90:
            scan_bools[2] = True  # Left
        elif move == 270:
            scan_bools[3] = True  # Right
        # else:
        #     scan_bools[1] = True  # Reverse
    
    return scan_bools

#####

def move_Cutebot(new_move, orientation, new_orientation=None):
        
    match new_move:
        case 0:
            # CutebotPro.distanceRunning(CutebotProOrientation.Advance, 5, CutebotProDistanceUnits.Cm)
            orientation = new_orientation if new_orientation is not None else orientation
            pass
        case 180:
            # CutebotPro.distanceRunning(CutebotProOrientation.Retreat, 5, CutebotProDistanceUnits.Cm)
            # orientation = (orientation + 180) % 360  # Update orientation after reverse
            orientation = new_orientation if new_orientation is not None else (orientation + 180) % 360  # Update after reverse
            pass
        case 90:
            # CutebotPro.angleRunning(CutebotProWheel.LeftWheel, -45, CutebotProAngleUnits.Angle)
            # CutebotPro.angleRunning(CutebotProWheel.RightWheel, 45, CutebotProAngleUnits.Angle)
            # CutebotPro.distanceRunning(CutebotProOrientation.Advance, 5, CutebotProDistanceUnits.Cm)
            orientation = new_orientation if new_orientation is not None else (orientation + 90) % 360  # Update after left turn
        case 270:
            # CutebotPro.angleRunning(CutebotProWheel.LeftWheel, 45, CutebotProAngleUnits.Angle)
            # CutebotPro.angleRunning(CutebotProWheel.RightWheel, -45, CutebotProAngleUnits.Angle)
            # CutebotPro.distanceRunning(CutebotProOrientation.Advance, 5, CutebotProDistanceUnits.Cm)
            orientation = new_orientation if new_orientation is not None else (orientation - 90) % 360  # Update after right turn
        case _:
            raise ValueError("Unrecognized move: {}".format(new_move))
        
    return orientation

#####

def on_forever(maze, pos, goal, path, moves_taken, orientation, epsilon, ph_array):
    reverse_counter = 0
    while pos != goal:
        # Scan environment
        scan_bools = scan_environment_test(pos, maze, orientation)
           
        # Determine viable moves based on scan_bools and path followed so far
        viable_moves = get_viable_moves(scan_bools, pos, path, orientation)
        
        # Determine next move using pheromone levels and epsilon-greedy strategy
        if viable_moves:  # If there are viable moves
            reverse_counter = 0  # Reset reverse counter
            new_move = select_move(viable_moves, epsilon, ph_array, pos, orientation)
            pos = step(new_move, orientation, pos)
            orientation = move_Cutebot(new_move, orientation)
            maze.display(pos)
            path.append(pos)
            moves_taken.append(new_move)
            if goal == pos:
                deposit_amount = 10  # Increase pheromone deposit amount when goal is reached
            else:
                deposit_amount = 1
            ph_array = drop_pheromones_single(pos, len(path), deposit_amount, ph_array)  # Drop pheromone at the new position
        else: 
            # If no viable moves, pop two steps from path and backtrack
            if path:
                pos = path[-2 - reverse_counter]
                new_orientation, reverse_counter = reverse_last_move(moves_taken, reverse_counter, orientation)
                orientation = move_Cutebot(180, orientation, new_orientation)  # Reverse move
                moves_taken.append(180)  # Record the reverse move
                maze.display(pos)
                path.append(pos)
            else:
                raise ValueError("No viable moves and path is empty. Cannot continue.")
    # ph_array = drop_pheromones(path, ph_array, deposit_amount)  # Drop pheromone at the goal position
    best_path = remove_backtracks(path)
    print("Best path found:", best_path)
    
    return best_path, ph_array

# basic.forever(on_forever)

##########

if __name__ == "__main__":
    # Initialize the maze and start the algorithm
    n_iterations = 100
    maze = WalledMaze((width, height), tuple(start), tuple(goal))
    maze.generate_maze()
    # ph_array = [[init_pher for _ in range(width)] for _ in range(height)]  # Reset pheromone array    
    for _ in range(n_iterations):
        # Reset position, path, and moves taken for each iteration
        pos = start.copy()
        path = [start.copy()]
        moves_taken = []
        orientation = 0  # Reset orientation to East
        # Run the maze solving algorithm
        best_path, ph_array = on_forever(maze, pos, goal, path, moves_taken, orientation, epsilon, ph_array)
        print("Iteration completed.")
        print("Length of path:", len(best_path))
    
    # Print the final path taken
    print("Final path taken:", path)
    print("Moves taken:", moves_taken)
    print("Final position:", pos)

##########