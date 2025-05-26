import numpy as np
import random
import math

#####
# Local imports
from maze_solver.solver.moves import Moves
from maze_solver.solver.state import State
from maze_solver.maze_generator.maze import Maze, WalledMaze

##################################################################

class Ant:
    
    def __init__(self, i, world):
        """
        Constructor for the Ant class.
        Parameters:
        - i (int): Index of the ant.
        - world (ACO): An instance of the ACO object containing the information about the maze and its pheromone levels.
        - start (State): The starting position of the ant in the maze.
        """
        self.ant_id = i  # ID/index of the ant
        self._init_ant(world)  # Initialise the ant's state
        
    #####
            
    def _init_ant(self, world):
        """
        Resets the ant's state for a new tour.
        Parameters:
        - world (ACO): An instance of the ACO object containing the information about the maze and its pheromone levels.
        """
        # Initialise instance variables parameters
        self.path = []
        self.path_length = 0
        self.unexplored = []
        self.transition_probs = []
        self.reachable = []  # reachable states from the current position (one move per timestep)
        self.allowed_moves = [Moves.UP, Moves.DOWN, Moves.LEFT, Moves.RIGHT]  # allowed moves in the maze
        self.possible_states = []  # the intersection of the unexplored and reachable states
        
        # Set other parameters to input values
        self.pos = world.start
        self.path = [world.start]
        self.unexplored = [s for s in world.states if s != world.start]
        self.goal = world.end
    
    #####
    
    @staticmethod
    def scan_environment(pos, world, reachable, allowed_moves):
        """
        Instructs ant to scan the environment to know which of the states are obstruction-free. 
        Parameters:
        - pos (State): The current position of the ant.
        - world (ACO): An instance of the ACO object containing the information about the maze and its pheromone levels.
        - allowed_moves (list): The list of allowed moves for the ant.
        - reachable (list): The list of reachable states from the current position.
        Returns:
        - filtered (list): The list of states that are obstruction-free.
        - filtered_moves (list): The list of moves that are obstruction-free.
        """
        maze_row, maze_col = Maze.to_maze_coords(pos, world.maze.size)  # Convert to maze coordinates
        filtered = []
        filtered_moves = []
        for move, state in zip(allowed_moves, reachable):  # Iterate over allowed moves & reachable states
            r, c = maze_row + move.dx, maze_col + move.dy
            if world.maze.grid[r][c] == 0:  # Check if the state is obstruction-free
                filtered.append(state)
                filtered_moves.append(move)
        
        return filtered, filtered_moves
         
    #####
    
    def set_reachable(self, world):
        """
        Sets the states that are reachable from the current position of the ant based on adjacency and obstructions. 
        Parameters:
        - world (ACO): An instance of the ACO object containing the information about the maze 
        and its pheromone levels.
        """
        # re-initialise transition probabilities at each timestep
        self.transition_probs = []  

        # Find the possible states and allowed moves
        self.possible_states, self.reachable, self.allowed_moves = Ant.find_reachable(self.pos, world, 
                                                                                      self.unexplored)
        # Update possible states
        # if backtrack:  # Don't include the most recent move
        #     self.possible_states = self.reachable
        #     self.possible_states.remove(self.path[-2])  # remove the last position to not get stuck again   
        #     self.possible_states = list(set(self.possible_states) & set(self.unexplored))         
        # else:

    #####
    
    @staticmethod
    def find_reachable(pos, world, unexplored=None):
        """
        Returns the states that are reachable from the current position of the ant.
        Parameters:
        - pos (State): The current position of the ant.
        - world (ACO): An instance of the ACO object containing the information about the maze and its pheromone levels.
        - unexplored (list): The list of unexplored states.
        Returns:
        - possible_states (Optional, list): The list of states that are reachable from the current position and unexplored.
        - reachable (list): The list of states that are reachable from the current position.
        - allowed_moves (list): The list of moves that are allowed in the maze.
        """
        reachable = []  # re-initialise reachable states at each timestep
        allowed_moves = []  # filtered moves based on the maze grid. Initialise.
        
        for move in Moves:
            new_x = pos.x + move.dx
            new_y = pos.y + move.dy
            new_idx = State.coords_to_idx(new_x, new_y, world.maze.size)
            
            # Check if the new position is within the grid dimensions
            if not (new_idx < 0 or new_idx >= world.num_states or (any(k < 0 for k in [new_x, new_y]))):
                allowed_moves.append(move)
                new_pos = State(pos.x + move.dx, pos.y + move.dy, new_idx)
                reachable.append(new_pos)
        
        # Scan the environment to check for obstructions
        reachable, allowed_moves = Ant.scan_environment(pos, world, reachable, allowed_moves)
        
        if unexplored is not None:
            possible_states = list(set(reachable) & set(unexplored))
            return possible_states, reachable, allowed_moves
        else:
            return reachable  # generic call from outside MP_tours
    
    #####
    
    def step(self, next_pos):
        """
        Moves the ant to the next position.
        Parameters:
        - next_pos (State): The next position of the ant.
        """
        self.pos = next_pos
        self.path.append(self.pos)
        
        # Remove the current position from unexplored states
        if self.pos in self.unexplored:
            self.unexplored.remove(self.pos)
        else:
            # print(f"Warning: {self.pos} is not in unexplored states.")
            pass
        
    #####
    
    def update_transition_prob(self, world, state, normalise=False):
        """
        Obtains the transition probability to a given state from the world's routing table. 
        Parameters:
        - world (ACO): An instance of the ACO object containing the information about the maze and its pheromone levels.
        - state (State): The state to which the transition probability is calculated.
        Returns:
        trans_prob (float): The transition probability to the given state.
        """
        # Obtain raw transition weight from routing table
        raw = world.routing_table[self.pos.index][state.index]
        # If requested, normalise by sum over all possible next states
        if normalise:
            denom = sum(world.routing_table[self.pos.index][c.index] for c in self.possible_states)
            # Avoid division by zero
            denom = denom if denom > 0 else world.epsilon
            raw = raw / float(denom)
        # Store and return the transition probability
        self.transition_probs.append(raw)
        return raw
    
    #####

    @staticmethod
    def calc_desirability(world, state_1, state_2):
        """
        Calculates the desirability of a state based on pheromone and attractiveness (optional).
        Parameters:
        - world (ACO): An instance of the ACO object containing the information about the maze and its pheromone levels.
        - state_1 (State): The first state.
        - state_2 (State): The second state.
        Returns:
        desirability (float): The desirability of the state.
        """    
        # desirability = (math.pow(world.pheromone[state_1.index][state_2.index], world.alpha)) * \
        #     (math.pow(world.attractiveness[state_1.index][state_2.index], world.beta))
        
        # as the attractiveness effect for adjacent states is none (based on distance), we can ignore it
        desirability = (math.pow(world.pheromone[state_1.index][state_2.index], world.alpha))
    
        return desirability
    
    #####
    
    @property
    def reached_goal(self) -> bool:
        """Check if the ant's current position matches the goal."""
        return self.pos == self.goal
    
    ##### 
    
    @property
    def pretty_path(self) -> list:
        """
        Pretty print (get) the path taken by the ant.
        """
        pretty_path = []
        for state in self.path:
            pretty_path.append((state.x, state.y))
        
        return pretty_path
        
    #####
    
    # def calc_transition_prob(self, world, state):
    #     """
    #     Calculates the transition probability to a given state, based on all the other reachable states. 
    #     Parameters:
    #     - world (ACO): An instance of the ACO object containing the information about the maze and its pheromone levels.
    #     - state (State): The state to which the transition probability is calculated.
    #     Returns:
    #     rel_desirability (float): The relative desirability of the transition to the given state.
    #     """
    #     # Compute desirability for the chosen state
    #     numer = Ant.calc_desirability(world, self.pos, state)
    #     # Sum desirabilities across all possible next states
    #     denom = sum(Ant.calc_desirability(world, self.pos, c) for c in self.possible_states)
    #     # Avoid division by zero
    #     denom = denom if denom > 0 else world.epsilon
    #     # Return the relative transition probability
    #     return numer / denom
        
    ##### 

    # def calc_path_length(self):
    #     """
    #     Calculates the length of the path taken by the ant using the Euclidean distance formula.
    #     Returns:
    #     sum_dist (float): The total distance of the path.
    #     """
    #     sum_dist = 0.00
    #     for i in range(0, len(self.path)):
    #         try:
    #             distance =  math.sqrt(math.pow((self.path[i].x - self.path[i+1].x), 2.0) + \
    #                 math.pow((self.path[i].y - self.path[i+1].y), 2.0))            
    #             sum_dist = sum_dist + distance
    #             self.path_length = sum_dist
    #         except:
    #             return sum_dist
    
    #####
    
##################################################################
