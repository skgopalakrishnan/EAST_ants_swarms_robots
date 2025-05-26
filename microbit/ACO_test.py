import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm

random.seed(0)  # For reproducibility

##################################################################

class Maze():
    def __init__(self, size, start, end):
        """
        Initializes the maze.
        - width: number of cells horizontally
        - height: number of cells vertically
        - start: tuple of (row, col) for start cell
        - end: tuple of (row, col) for end cell
        """
        # Validate inputs
        assert len(size) == 2, "Size must be a tuple of two integers."
        assert size[0] > 0 and size[1] > 0, "Size must be positive integers."
        
        # Initialize the maze size and grid
        self.height, self.width = size
        self.size = size
    
        # To track which cells have been visited in the maze generation
        self.visited = np.zeros((self.height, self.width), dtype=bool)
        
        # Start and end positions
        assert 0 <= start[0] < self.height and 0 <= start[1] < self.width, "Start cell out of bounds."
        assert 0 <= end[0] < self.height and 0 <= end[1] < self.width, "End cell out of bounds."
        self.start = start
        self.end = end
    
    #####  
    
    def display(self, ant_pos=None, reachable=None, ax=None, clear_first=True, 
                pheromones=None, final_path=None, save_fig=False, **kwargs):
        """Visualize the maze using matplotlib, optionally overlaying the ant's position and reachable states.
        if ax is provided, it will be used for plotting; otherwise, a new figure will be created.
        Parameters:
        - ant: tuple of (row, col) for the ant's position
        - reachable: list of tuples of (row, col) for reachable states
        - ax: matplotlib axis object for plotting
        - clear_first: whether to clear the current figure before plotting
        - pheromones: optional pheromone levels to overlay on the maze
        - final_path: list of tuples of (row, col) for the final path
        - save_fig: whether to save the figure as an image
        - kwargs: additional keyword arguments for customization
        """
        
        # Whether to reuse figure/create a new figure
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 5))
        else:
            fig = ax.figure
            if clear_first:
                ax.cla()
                
        # Plot the maze grid
        if pheromones is None:
            ax.imshow(self.grid, cmap='binary')
            
        ax.set_xticks([]); ax.set_yticks([])
        
        # Mark start and end positions
        start_row, start_col = Maze.to_maze_coords(self.start, self.size)
        ax.plot(start_col, start_row, 'go', markersize=10)  # Start in green
            
        end_row, end_col = Maze.to_maze_coords(self.end, self.size)
        ax.plot(end_col, end_row, 'bo', markersize=10)  # End in blue
        
        ax.set_aspect('equal', adjustable='box')  # Keep the aspect ratio
        ax.set_xlim(-0.5, self.grid.shape[1] - 0.5)
        ax.set_ylim(self.grid.shape[0] - 0.5, -0.5)  # Invert y-axis to match the grid
        
        if ant_pos is not None:
            # Mark the ant's position
            ant_row, ant_col = self.to_maze_coords(ant_pos, self.size)
            ax.plot(ant_col, ant_row, 'ro', markersize=10)

            # List reachable states
            if reachable is not None:
                for state in reachable:
                    row, col = Maze.to_maze_coords(state, self.size)
                    ax.plot(col, row, 'yx', markersize=10)
            ax.set_title("Maze with Ant")
         
        else:
            ax.set_title("Maze")
            
        if pheromones is not None:
            raise NotImplementedError("Pheromone visualization is not implemented yet.")
            # # Instead of plotting pheromone[i][j] directly at cell centers,
            # # aggregate all outgoing edges for each cell to show "how likely" it is traveled.
            # num_cells = self.height * self.width
            # heatmap = np.zeros_like(self.grid, dtype=float)
            # for idx in range(num_cells):
            #     # Convert flat index to (x, y)
            #     x = idx // self.width
            #     y = idx % self.width
            #     r, c = Maze.to_maze_coords((x, y), self.size)
            #     # Sum outgoing edges' pheromone
            #     cell_pheromone = np.sum(pheromones[idx])
            #     heatmap[r, c] = cell_pheromone
            # max_val = heatmap.max()
            # if max_val > 0:
            #     heatmap /= max_val
            # # heatmap = np.sqrt(heatmap)
            # # heatmap = 1 - heatmap
            # ax.imshow(heatmap, cmap='hot', alpha=0.6, interpolation='nearest')

        if ax is None:  # if no ax is provided, we need to show the figure
            plt.show()
        else:  # if ax is provided, we need to draw on the canvas/overlay
            fig.canvas.draw()
        
        if final_path is not None:
            # Plot the final path
            for i in range(len(final_path) - 1):
                start = Maze.to_maze_coords(final_path[i], self.size)
                end = Maze.to_maze_coords(final_path[i + 1], self.size)
                ax.plot([start[1], end[1]], [start[0], end[0]], 'g-', linewidth=2)
                ax.plot(end[1], end[0], 'ro', markersize=10)  # Mark the end of the path
        
            # Set the title and labels
            ax.set_title("Maze with Ant Path")
        
        if save_fig:
            # Save the figure as an image
            fig.savefig("example_path.png", bbox_inches='tight')
        
        return fig, ax
    
    #####
        
    def generate_maze(self):
        """Generates the maze using a specific algorithm."""
        raise NotImplementedError("Implement a specific maze. This is the base class.")
        pass
    
    #####
     
    def _carve_path(self):
        """Carves a direct path from start to end to ensure solvability."""
        current = self.start
        # Mark the start cell as visited
        self.visited[current[0]][current[1]] = True
        while current != self.end:
            row, col = current
            end_row, end_col = self.end
            if random.random() < 0.5:  # 50% chance to prioritize row or column movement
                dr, dc = 0, 1 if col < end_col else -1  # Move column-first
            else:
                dr, dc = 1 if row < end_row else -1, 0  # Move row-first
            new_row, new_col = (row + dr, col + dc)
            # Remove the wall between current and new cell
            wall_row = 2*row + 1 + dr  # since new_row = row + dr
            wall_col = 2*col + 1 + dc
            self.grid[wall_row][wall_col] = 0  # remove the wall between the open cells
            # Mark as visited
            self.visited[new_row][new_col] = True            
            # Update current position and loop
            current = (new_row, new_col)
    
    @staticmethod
    def to_maze_coords(state, grid_dimensions):
        """
        Convert a state (x, y) to maze coordinates (row, col) which has gaps for walls.
        Parameters:
        - state: tuple of (x, y) coordinates
        - grid_dimensions: tuple of (height, width) of the maze
        Returns:
        row, col: coordinates in the maze grid with walls...
        """
        x, y = state
        row = 2 * x + 1
        col = 2 * y + 1
        return row, col
    
    #####
     
    def save(self, filename, save_img=False):
        """Saves the maze to a file.
        Parameters:
        - filename: name of the file to save the maze
        - save_img: whether to save the maze as an image.
        """
        if filename is None:
            filename = 'maze'
        # Ensure the filename has a .npy extension
        if not filename.endswith('.npy'):
            filename += '.npy'
        np.save(filename, self.grid)  # can only save the grid
        print(f"Maze saved to {filename}.npy")

        # Save the maze as an image if requested
        if save_img:
            plt.imshow(self.grid, cmap='binary')
            plt.xticks([]), plt.yticks([])
            plt.title("Maze")
            img_filename = filename.replace('.npy', '.png')
            plt.savefig(img_filename)
            print(f"Maze image saved to {img_filename}")
         
    #####        

    def load(self, filename):
        """Loads the maze from a file."""
        if filename is None:
            filename = 'maze'  # assume a default filename
        # Ensure the filename has a .npy extension
        if not filename.endswith('.npy'):
            filename += '.npy'
        try:
            self.grid = np.load(filename)
            print(f"Maze loaded from {filename}.npy")
            self.height, self.width = self.grid.shape  # update height and width from loaded grid
        except FileNotFoundError:
            print(f"File {filename}.npy not found. Please check the filename and try again.")
        
##########

class WalledMaze(Maze):
    def __init__(self, size, start, end):
        """
        Initializes the maze with continuous walls. 0 in grid means a passage, 1 means a wall.
        - size: tuple of (height, width)
        """
        super().__init__(size, start, end)
        self.grid = np.ones((2 * self.height + 1, 2 * self.width + 1), dtype=int)  # all walls to start
        # Mark the cell positions (odd indices) as unvisited/empty (0 means a passage will be carved later)
        for i in range(self.height):
            for j in range(self.width):
                self.grid[2*i+1][2*j+1] = 0  # looks like a chessboard. 0 for every open cell.
    
    #####
    
    def generate_maze(self):
        """Generates the walled maze using recursive backtracking."""
        # Carve a guaranteed path from start to end
        self._carve_path()
        print("Carved path from start to end.")
        
        # Reset the visited cells on the carved path
        self.visited = np.zeros((self.height, self.width), dtype=bool)
        
        # Carve random passages in the maze from the start cell
        self._dfs(self.start[0], self.start[1])
        print("Carved random passages in the maze.")
    
    #####    
        
    def _dfs(self, row, col):
        """Recursive DFS to carve passages in the maze."""
        self.visited[row, col] = True  # mark the start as visited
        
        # Directions: up, down, left, right
        directions = [(0, 1), (0, -1), (-1, 0), (1, 0)]
        
        # Randomize the order of directions
        random.shuffle(directions)
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            
            # Check if the new cell is within bounds and not visited
            if (0 <= new_row < self.height) and (0 <= new_col < self.width) and not (self.visited[new_row, new_col]):                
                # Carve a passage between the current cell and the new cell
                wall_row = row + new_row + 1  # converting to grid indices. 
                wall_col = col + new_col + 1  # same for column
                self.grid[wall_row][wall_col] = 0  # remove the wall
                # print("Visited:", (new_row, new_col))
                # Recur for the new cell
                self._dfs(new_row, new_col)
            # After all directions are explored, backtrack
            
##########

def get_viable_moves(scan_bools, pos, path, orientation):
    viable_moves = []
    # neighbours = find_neighbours(pos)
    if scan_bools[0]:
        if step(0, orientation, pos) not in path:  # Forward
            viable_moves.append(0)
    elif scan_bools[1]:
        if step(180, orientation, pos) not in path:  # Backward
            viable_moves.append(180)
    if scan_bools[2]:
        if step(90, orientation, pos) not in path:  # Left
            viable_moves.append(90)
    if scan_bools[3]:
        if step(270, orientation, pos) not in path:  # Right
            viable_moves.append(270)
    return viable_moves 
    
##########

# def find_neighbours(pos):
#     neighbours = []
#     # Assuming pos is a tuple (row, col)
#     row, col = pos
#     # Check all four possible directions
#     for dr, dc in [(0, 1), (-1, 0), (0, 1), (0, -1)]:
#         new_row, new_col = row + dr, col + dc
#         neighbours.append((new_row, new_col))
#     return neighbours
    
    
##########

# def reverse_move(pos, move, orientation):

##########

# Epsilon-greedy selection of moves based on pheromone levels and exploration rate.
def select_move(viable_moves, epsilon, ph_array, pos, orientation):
    if len(viable_moves) == 1:
        # If only one viable move, return it
        return viable_moves[0]   
    elif random.random() < epsilon:
        # Explore: choose a random move from viable moves
        return random.choice(viable_moves)
    else:
        new_pos_options = [step(move, orientation, pos) for move in viable_moves]
        # Calculate pheromone levels for each viable move
        pheromone_levels = [ph_array[new_pos[0]][new_pos[1]] for new_pos in new_pos_options]
        # Normalize pheromone levels to probabilities
        total_pheromone = sum(pheromone_levels)
        probabilities = [ph / total_pheromone for ph in pheromone_levels]
        # Select a move based on the probabilities
        return random.choices(viable_moves, weights=probabilities, k=1)[0]

##########

def reverse_last_move(moves_taken, reverse_counter, orientation):
    
    last_move = moves_taken[-1 - reverse_counter]
    if last_move == 90:
        new_orientation = (orientation - 90) % 360 if len(moves_taken) > 1 else None
    elif last_move == 270:
        new_orientation = (orientation + 90) % 360 if len(moves_taken) > 1 else None
    else: 
        new_orientation = None
    reverse_counter += 2
    
    return new_orientation, reverse_counter

##########

def step(new_move, orientation, pos):

    def calc_delta(new_move, orientation):
        if orientation == 0:
            match new_move:
                case 0:  # forward
                    return 0, 1
                case 180:
                    return 0, -1
                case 90:
                    return -1, 0
                case 270:
                    return +1, 0
        elif orientation == 90:
            match new_move:
                case 0:
                    return -1, 0
                case 180:
                    return 1, 0
                case 90:
                    return 0, -1
                case 270:
                    return 0, 1
        elif orientation == 180:
            match new_move:
                case 0:
                    return 0, -1
                case 180:
                    return 0,  1
                case 90:
                    return 1,  0
                case 270:
                    return -1, 0
        elif orientation == 270:
            match new_move:
                case 0:
                    return 1, 0
                case 180:
                    return -1, 0
                case 90:
                    return 0, 1
                case 270:
                    return 0, -1
            
    delta_x, delta_y = calc_delta(new_move, orientation)
    new_pos = [pos[0] + delta_x, pos[1] + delta_y]
    return new_pos

##########

def remove_backtracks(path):
    """
    # Remove loops/backtracked segments from the path of the ant.
    This is done by checking if the current state is already in the cleaned path.
    Parameters:
    - a (Ant): The ant whose path needs to be updated.
    """
    cleaned_path = []
    detected = False  # Flag to indicate if a loop was detected
    for state in path:
        if state in cleaned_path:
            # Found a loop, backtrack by popping until we return to this state
            while cleaned_path and cleaned_path[-1] != state:
                cleaned_path.pop()
                detected = True
        else:
            cleaned_path.append(state)

    # Update the ant's path
    path = cleaned_path
    # print("Found a loop. Cleaned it.")

    return path

##########

def drop_pheromones_single(pos, path_length, deposit_amount, ph_array):
    r, c = pos
    delta_pher = deposit_amount / (path_length**1)
    ph_array[r][c] += delta_pher  # update pheromone level
    return ph_array

##########
    
def drop_pheromone(path, ph_array, deposit_amount):
    for k in range(0, len(path) - 1):
        r, c, = path[k]
        delta_pher = deposit_amount / (len(path)**1)  # shorter paths get more pheromone
        ph_array[r][c] += delta_pher  # update pheromone level
    return ph_array

##########