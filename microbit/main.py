width = 5
height = 5
start = [0, 0]
goal = [4, 4]
init_pher = 0.001  # Initial pheromone level - prevent division by zero
deposit_amount = 1  # Amount of pheromone to deposit
epsilon = 0.1
pos = start
scan_bools = [False, False, False, False]
allowed_moves = [0, 90, 180, 270]  # Representing moves as angles
orientation = 0  # 0: East, 90: North, 180: West, 270: South
num_reverses = 0  # Counter for reverses
nStates = width * height
ph_array = []
for i in range(nStates):
    ph_array.push(init_pher)

path_length = 1


# helpers to map (r,c) <-> flat indices
def xy_to_state(r, c):
    return r * width + c

def idx(i, j):
    return i * nStates + j


def on_forever():
    while pos != goal:
        # Scan environment
        CutebotPro.trackbit_state_value()
        if CutebotPro.get_grayscale_sensor_state(TrackbitStateType.TRACKING_STATE_1):

            scan_bools[0] = True
        elif CutebotPro.get_grayscale_sensor_state(TrackbitStateType.TRACKING_STATE_9):

            scan_bools[4] = True
        elif CutebotPro.get_grayscale_sensor_state(TrackbitStateType.TRACKING_STATE_13):
            CutebotPro.pwm_cruise_control(-10, -10)
        elif CutebotPro.get_grayscale_sensor_state(TrackbitStateType.TRACKING_STATE_0):
            CutebotPro.pwm_cruise_control(10, 10)
            CutebotPro.distance_running(CutebotProOrientation.RETREAT, 3, CutebotProDistanceUnits.CM)
            CutebotPro.distance_running(CutebotProOrientation.ADVANCE, 0, CutebotProDistanceUnits.CM)
            CutebotPro.angle_running(CutebotProWheel.LEFT_WHEEL, 0, CutebotProAngleUnits.ANGLE)
basic.forever(on_forever)


#####

def scanEnvironment():
    scan_bools = [False, False, False, False]  # Reset scan_bools
    CutebotPro.trackbit_state_value()
    if CutebotPro.get_grayscale_sensor_state(TrackbitStateType.TRACKING_STATE_1):
        # Straight path
        scan_bools[0] = True
    elif CutebotPro.get_grayscale_sensor_state(TrackbitStateType.TRACKING_STATE_9):
        # Left turn
        scan_bools[2] = True
    elif CutebotPro.get_grayscale_sensor_state(TrackbitStateType.TRACKING_STATE_13):
        # Right turn
        scan_bools[3] = True
    elif CutebotPro.get_grayscale_sensor_state(TrackbitStateType.TRACKING_STATE_0):
        # Reverse / No forward path
        scan_bools[1] = True
    return scan_bools

#####

def moveCutebot(new_move, orientation, new_orientation=None):
    if new_move == 0:
        orientation = new_orientation if new_orientation is not None else orientation
        CutebotPro.distanceRunning(CutebotProOrientation.Advance, 5, CutebotProDistanceUnits.Cm)
    elif new_move == 180:
        CutebotPro.distanceRunning(CutebotProOrientation.Retreat, 5, CutebotProDistanceUnits.Cm)
        # orientation = (orientation + 180) % 360  # Update orientation after reverse
        orientation = new_orientation if new_orientation is not None else (orientation + 180) % 360  # Update after reverse
    elif new_move == 90:
        CutebotPro.angleRunning(CutebotProWheel.LeftWheel, -45, CutebotProAngleUnits.Angle)
        CutebotPro.angleRunning(CutebotProWheel.RightWheel, 45, CutebotProAngleUnits.Angle)
        CutebotPro.distanceRunning(CutebotProOrientation.Advance, 5, CutebotProDistanceUnits.Cm)
        orientation = new_orientation if new_orientation is not None else (orientation + 90) % 360  # Update after left turn
    elif new_move == 270:
        CutebotPro.angleRunning(CutebotProWheel.LeftWheel, 45, CutebotProAngleUnits.Angle)
        CutebotPro.angleRunning(CutebotProWheel.RightWheel, -45, CutebotProAngleUnits.Angle)
        CutebotPro.distanceRunning(CutebotProOrientation.Advance, 5, CutebotProDistanceUnits.Cm)
        orientation = new_orientation if new_orientation is not None else (orientation - 90) % 360  # Update after right turn
    else:
        raise ValueError("Unrecognized move!")
    
    return orientation

#####


def solve_maze(maze, pos, goal, orientation, epsilon, ph_array):

    path = []
    path.push(xy_to_state(0, 0))
    moves_taken = []
    reverse_counter = 0

    while pos != goal:
        

        # Scan environment
        scan_bools = scanEnvironment()
           
        # Determine viable moves based on scan_bools, path followed so far, and current positioning
        viable_moves = ACO.getViableMoves(scan_bools, pos, path, orientation)
        
        # Determine next move using pheromone levels and epsilon-greedy strategy
        if viable_moves:  # If there are viable moves
            new_move = ACO.selectMove(viable_moves, epsilon, ph_array, pos, orientation)
            
            # Update the target position
            pos = ACO.step(new_move, orientation, pos)
            
            # Update orientation and move the Cutebot
            orientation = moveCutebot(new_move, orientation)
            ACO.showOrientation(orientation)  # Show the current orientation
            
            # Update the lists storing the path and moves taken
            path.push(pos)
            moves_taken.push(new_move)
            path_length = path_length + 1

            # Deposit the pheromones to convey signal to the other ants
            if goal == pos:
                deposit_amount = 10  # Increase pheromone deposit amount when goal is reached
            else:
                deposit_amount = 1
            ph_array = ACO.dropPheromone(path[-1], pos, path_length, ph_array, deposit_amount)  # Drop pheromone at the new position
            
            # Miscellaneous updates
            reverse_counter = 0
        
        else: # If no viable moves, pop two steps from path and backtrack
            reverse_i = len(path) - 2 - reverse_counter
            
            if reverse_i < 0: break  # nowhere to back up
            pos = path[reverse_i]
            
            # Find the new orientation after backtracking
            new_orientation, reverse_counter = ACO.reverse_last_move(moves_taken, reverse_counter, orientation)
            
            # Update orientation and move the Cutebot backwards
            orientation = moveCutebot(180, orientation, new_orientation)  # Reverse move
            
            # Updat the lists storing the path and moves taken
            path.push(pos)
            moves_taken.push(180)  # Record the reverse move

    # Buzz to signal the goal is reached
    return ph_array  # Return the updated pheromone array