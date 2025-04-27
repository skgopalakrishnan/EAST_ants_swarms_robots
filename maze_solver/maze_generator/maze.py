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
                pheromones=None, final_path=None, **kwargs):
        """Visualize the maze using matplotlib, optionally overlaying the ant's position and reachable states.
        if ax is provided, it will be used for plotting; otherwise, a new figure will be created.
        Parameters:
        - ant: tuple of (row, col) for the ant's position
        - reachable: list of tuples of (row, col) for reachable states
        - ax: matplotlib axis object for plotting
        - clear_first: whether to clear the current figure before plotting
        - pheromones: optional pheromone levels to overlay on the maze
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
            
##################################################################

# class ObstacleMaze(Maze):
    #     def __init__(self, size, start, end, obstacle_density=0.2):
    #         """
    #         Initializes the maze with obstacles placed at random locations.
    #         size: tuple of (height, width)
    #         obstacle_density: fraction of cells to fill with obstacles.
    #         """
    #         raise NotImplementedError("ObstacleMaze is not implemented yet.")
    #         super().__init__(size, start, end)
    #         self.density = obstacle_density
    #         # Initialize with no walls – an empty grid.
    #         self.grid = np.zeros((2 * self.height + 1, 2 * self.width + 1), dtype=int)
    #         # Ensure that cell positions (odd indices) are initially open.
    #         for i in range(self.height):
    #             for j in range(self.width):
    #                 self.grid[2*i+1][2*j+1] = 0  
            
    #     def generate_maze(self):
    #         """Generates the maze by placing rectangular obstacles of random sizes
    #         and carving a guaranteed path from start to end.
    #         DFS is not applied after obstacle placement.
    #         """
    #         num_cells = self.height * self.width
    #         num_obstacles = int(self.density * num_cells)
            
    #         for _ in range(num_obstacles):
    #             # Choose a random obstacle size in cell units (between 1 and 3, or maximum available)
    #             obs_h = random.randint(1, min(2, self.height))
    #             obs_w = random.randint(1, min(2, self.width))
            
    #             # Choose a random top-left cell so that the obstacle fits within the maze cells
    #             max_row = self.height - obs_h
    #             max_col = self.width - obs_w
    #             top_left_row = random.randint(0, max_row)
    #             top_left_col = random.randint(0, max_col)
                
    #             # Convert cell coordinates to grid coordinates
    #             row_start = 2 * top_left_row
    #             row_end = 2 * (top_left_row + obs_h) + 1  # +1 because slice end is exclusive
    #             col_start = 2 * top_left_col
    #             col_end = 2 * (top_left_col + obs_w) + 1
                
    #             # Place the obstacle by marking the corresponding region as walls (value 1)
    #             self.grid[row_start:row_end, col_start:col_end] = 1

    #         # Ensure that the start and end cells remain open
    #         start_row, start_col = 2 * self.start[0] + 1, 2 * self.start[1] + 1
    #         end_row, end_col = 2 * self.end[0] + 1, 2 * self.end[1] + 1
    #         self.grid[start_row, start_col] = 0
    #         self.grid[end_row, end_col] = 0
            
    #         # Carve a guaranteed path from start to end (this may remove some obstacles along the path)
    #         self._carve_path()
    #         print("Carved guaranteed path from start to end in ObstacleMaze.")

    # def _carve_path(self):
    #     """Carves a direct path from start to end to ensure solvability."""
    #     current = self.start
    #     self.visited[current[0]][current[1]] = True
    #     # Force open the starting cell (in case an obstacle was placed there)
    #     start_cell_row, start_cell_col = 2 * self.start[0] + 1, 2 * self.start[1] + 1
    #     self.grid[start_cell_row][start_cell_col] = 0
        
    #     while current != self.end:
    #         row, col = current
    #         end_row, end_col = self.end
    #         dr = 0
    #         dc = 0
    #         if row < end_row:
    #             dr = 1
    #         elif row > end_row:
    #             dr = -1
    #         if col < end_col:
    #             dc = 1
    #         elif col > end_col:
    #             dc = -1
    #         # If both directions are possible, randomly choose one
    #         if dr != 0 and dc != 0:
    #             if random.choice([True, False]):
    #                 dc = 0
    #             else:
    #                 dr = 0
    #         new_row, new_col = row + dr, col + dc
    #         # Remove the wall between current and new cell
    #         wall_row = row + new_row + 1  # average position in grid coordinates
    #         wall_col = col + new_col + 1
    #         self.grid[wall_row][wall_col] = 0
    #         # Force open the new cell itself (this is the key change)
    #         cell_row = 2 * new_row + 1
    #         cell_col = 2 * new_col + 1
    #         self.grid[cell_row][cell_col] = 0
    #         self.visited[new_row][new_col] = True            
    #         current = (new_row, new_col)

##################################################################

if __name__ == "__main__":
    # Example usage
    maze_size = (5, 5)
    start = (0, 0)
    end = (4, 4)
    
    maze = WalledMaze(maze_size, start, end)
    maze.generate_maze()
    maze.display()
    
    # Save the maze
    maze.save("maze_example", save_img=True)
    
    # Load the maze
    loaded_maze = WalledMaze((0, 0), (0, 0))
    loaded_maze.load("maze_example.npy")
    loaded_maze.display()
    
##################################################################
