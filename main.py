import random

from maze_solver.solver.ACO import ACO
import math
import matplotlib.pyplot as plt
import random

##################################################################

if __name__ == "__main__":
    
    # Initialise the maze
    grid_dimensions = (5, 5)  # Dimensions of the grid
    num_states = grid_dimensions[0] * grid_dimensions[1]  # Total number of positions
        
    # Set up the ants and the simulation parameters
    num_steps = 1000  # Number of steps for the simulation
    n_ants = 1  # Number of ants
    initial_pheromone = 1.0  # Initial pheromone level
    
    # Create the ACO object solver
    aco = ACO(num_states, initial_pheromone=initial_pheromone, alpha=3,
              beta=3, epsilon=0.1, pheromone_deposit=2,
              evaporation_constant=0, early_stop_n=3, type_="maze")
    
    # Run the ACO algorithm to get the best path
    shortest_path, shorted_paths, shortest_path_lens = \
        aco.get_best_path(num_ants=n_ants, num_steps=num_steps) 
    
    # Print the best path
    print("Best path found by ACO:")
    for step in shortest_path:
        print(f"Step: {step}")
        
    # Plot convergence of shortest path length over steps
    plt.figure()
    plt.plot(range(1, len(shortest_path_lens) + 1), shortest_path_lens, marker='o')
    plt.xlabel("Step Number")
    plt.ylabel("Shortest Path Length")
    plt.title("Convergence of ACO Algorithm")
    plt.grid(True)
    # Save the plot
    plt.savefig("example_output.png")
    plt.show()

##################################################################
