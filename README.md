README v1
Repository containing the code related to schools outreach projects.

**maze_solver** : Package containing modules related to maze solving.
This package includes the following modules:

- `maze_solver`: Contains the main maze-solving algorithm and logic.
- `maze_generator`: Contains functions to generate mazes for testing and demonstration purposes.
- `visualizer`: Provides functions to visualize the maze and the solving process.

## Usage

```python
from maze_solver.solver.ACO import ACO
import matplotlib.pyplot as plt

# Initialise ACO for a 10×10 maze
grid_size = (10, 10)
num_states = grid_size[0] * grid_size[1]
aco = ACO(num_states,
          initial_pheromone=1.0, alpha=3, beta=3,
          epsilon=0.1, pheromone_deposit=2,
          evaporation_constant=0.6, early_stop_n=3,
          type_="maze")

# Run optimization
shortest_path, all_paths, path_lens = aco.get_best_path(num_ants=1, num_steps=100)

# Display result
print("Shortest path:", [(s.x, s.y) for s in shortest_path])

# Plot convergence
plt.plot(range(1, len(path_lens) + 1), path_lens, marker='o')
plt.xlabel("Step")
plt.ylabel("Path Length")
plt.title("ACO Convergence")
plt.grid(True)
plt.show()
```

To use the maze solver, import the `maze_solver` module and call the `solve_maze` function with a maze as input. The maze should be represented as a 2D list or array.

Example:

```python
from maze_generator import walled_maze
from maze_solver import depth_first_search
from visualiser import static_plot, animated_plot

#### Generate maze ####
# Custom maze
# maze = [
#     [0, 1, 0, 0],
#     [0, 1, 0, 1],
#     [0, 0, 0, 1],
#     [1, 1, 0, 0]
# ]
# Auto-generated maze
maze = walled_maze(rows=4, cols=4, encoding="binary")

#### Solve maze ####
start = (0, 0)  # indexing is positive x/y system (right, up)
end = (3, 3)
solution = depth_first_search(maze, start, end)

#### Visualise the obtained solution ####
print(solution)
static_plot(solution)
```