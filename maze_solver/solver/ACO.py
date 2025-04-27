# Standard library imports
import numpy as np
from multiprocessing import Process, Queue
import random
import os
import time

#####
# Local imports
from maze_solver.solver.state import State
from maze_solver.solver.ant import Ant
from maze_solver.solver.moves import Moves
from maze_solver.maze_generator.maze import WalledMaze

#####
random.seed(0)  # Set random seed for reproducibility

##################################################################

class ACO:
    def __init__(self, num_states, start=0, end=-1, initial_pheromone=1, alpha=1, beta=3, epsilon=0.01,
                pheromone_deposit=1, evaporation_constant=0.6, early_stop_n=5, type_="maze", **kwargs):
        """
        ACO class for solving the Traveling Salesman Problem (TSP) using Ant Colony Optimization.
        Parameters:
        - num_states (int): Number of cities in the maze.
        - initial_pheromone (float, optional): Initial pheromone concentration on all edges. Default 1.0
        - alpha (float, optional): Pheromone influence exponent in path selection probability. Default 1
        - beta (float, optional): Heuristic (distance) influence exponent in path selection. Default 3
        - epsilon (float, optional): Probability of exploration. Default 0.01
        - pheromone_deposit (float, optional): Pheromone deposit amount per ant. Default 1
        - evaporation_constant (float, optional): Pheromone evaporation rate (0-1). Default 0.6
        - type (str, optional): Type of problem to solve. Default "maze"
        """
        # Initialise parameters
        self.states = []
        self.shortest_paths = []
        self.shortest_path_lens = [] 
        self.shortest_path_len = -1
        self.early_stopping_n = early_stop_n
        
        # Set other parameters to input values
        self.evaporation_const = evaporation_constant
        self.pheromone_deposit = pheromone_deposit
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.initial_pheromone = initial_pheromone
        
        print("Adding the states to the ACO instance...")
        if type_ == "maze":
            if not ("width" in kwargs and "height" in kwargs):  # Assume square maze
                width =  int(np.sqrt(num_states))
                height = num_states // width
                assert num_states == width * height, "Number of states must be a perfect square for square maze"
            else:
                width = kwargs["width"]
                height = kwargs["height"]
            self._add_states(single=False, width=width, height=height)
        else:
            raise ValueError("Invalid type. Only 'maze' is supported for now.")

        # Set start and end states        
        self._set_start_end(start, end)
        print("Initialised ACO instance with start and end states at x-y positions {} and {} respectively.".format(start, end))

        # Generate the maze
        self.maze = WalledMaze((width, height), tuple(self.start), tuple(self.end))
        self.maze.generate_maze()
        self.fig, self.ax = self.maze.display()
        print("Maze generated.")
        
    #####   

    def _set_start_end(self, start, end):
        """
        Set the start and end states for the ACO instance.
        Parameters:
        - start (int): Index of the start state.
        - end (int): Index of the end state.
        """
        assert start != end, "Start and end states must be different"
        if start == 0:
            self.start = self.states[0]
        elif start == -1:
            self.start = self.states[-1]
        if end == 0:
            self.end = self.states[0]
        elif end == -1:
            self.end = self.states[-1]
        if (start not in [0, -1]) or (end not in [0, -1]):
            raise NotImplementedError("Start and end states must be either 0 or -1.")
        
    #####

    def _add_states(self, single=True, **kwargs):
        """
        Add states to the ACO instance.
        Parameters:
        - single (list): Whether to add a single state to the list or multiple states.
        - kwargs (dict): Additional arguments for state creation.
            - x (int): x-coordinate of the state.
            - y (int): y-coordinate of the state.
            - index (int, optional): Index of the state.
            - width (int): Width of the maze.
            - height (int): Height of the maze.
        """
        if single:
            raise NotImplementedError("Single state addition is not implemented yet.")
            # if "x" in kwargs and "y" in kwargs:
            #     x = kwargs.get("x")
            #     y = kwargs.get("y")
            #     index = kwargs.get("index", None)
            #     self.states.append(State(x, y, index))                
            # else:
            #     raise ValueError("x and y coordinates must be provided for single state.")
        else:
            width = kwargs.get("width")
            height = kwargs.get("height")
            states = [State(i, j, idx) for idx, (i, j) in enumerate(np.ndindex(height, width))]  # row-major order
            self.states.extend(states)

        # Initialize pheromone and routing table matrices
        self.num_states = len(self.states)
        print("Added {} states to the ACO instance.".format(self.num_states))
        print("Initialising pheromone and routing table matrices...")
        self._init_matrices()
        print("Initialised pheromone and routing table matrices.")
        
    #####
        
    def _init_matrices(self):
        """
        Initialize/Reinitialise the pheromone and routing table matrices.
        """
        self.routing_table = np.full((self.num_states, self.num_states), (1.00/(self.num_states - 1)))  # evenly distributed
        np.fill_diagonal(self.routing_table, 0.0)  # zero out self‐loops
        self.pheromone = np.full((self.num_states, self.num_states), self.initial_pheromone)  # initial pheromone
        # self.attractiveness = np.zeros((self.num_states, self.num_states))  # attractiveness matrix. Initialised to zero
        
    #####
    
    def get_best_path(self, num_ants=1, num_steps=1, multi_processing=False, **kwargs):    
        """
        Get the best path using Ant Colony Optimization.
        Parameters:
            num_ants (int, optional): Number of ants to use. Defaults to 1.
            num_steps (int, optional): Number of steps to take. Defaults to 1.
        Returns:
        output_path (list): The best path found by the ants.
        """
        # Reset the shortest paths and lengths
        self.shortest_paths = []
        self.shortest_paths_lens = []
        
        # Initialise ants and compute attraction matrix based on distances
        # self._calc_attraction()
        ants = []
        
        # Create ants
        for i in range(0, num_ants):
            ants.append(Ant(i, self))
        
        # Run ACO algorithm for the specified number of steps
        for step in range(0, num_steps):
            print("Step: ", step+1, " of ", num_steps)
            path_lens = []
            paths = []
            procs = []
            
            if multi_processing:
                # print("Running ACO in multiprocessing mode...")
            
                # Create a queue and start one process per ant whereby each ant follows a pheromone trail
                # and deposits pheromone on the path taken
                q = Queue()
                for a in ants:
                    p = Process(target=self._mp_tour, args=(a, q, True))
                    procs.append(p)
                    p.start()

                for p in procs:
                    p.join()

                ants = []  # recreate ants list to store the ants that have completed their tours
                while q.empty() == False:
                    ants.append(q.get())  # get the ant object from the queue ("put")
                    
            else:
                # print("Running ACO in single-threaded mode...")
                for a in ants:
                    self._mp_tour(a, None, multi_processing=False, display=True)  # changes made in-place
                    
            # Calculate the path length for each ant and update pheromone levels
            for a in ants:
                print("Ant: ", a.ant_id)
                path_len = len(a.path)
                path_lens.append(path_len)
                paths.append(a.path)
                print("Path length: ", path_len)
                
                # Remove backtracks from the path
                self._remove_backtracks(a)
                self.maze.display(ant_pos=a.pos, reachable=None, ax=self.ax, 
                                  clear_first=True, final_path=a.path)  # display cleaned final path
                
                # Update pheromone levels and routing table based on each ant's path
                self._update_pheromone(a)
                self._update_routing_table(a)

            self._evaporate_pheromone()  # evaporate pheromone levels once per iteration
            
            # TODO Display the maze with pheromone levels 
            # self.maze.display(ax=self.ax, pheromones=self.pheromone, clear_first=True)
            
            # Store best path and its length
            best_path_len = min(path_lens)
            best_path = paths[path_lens.index(best_path_len)]

            print("Step best path: ", best_path_len, " Step: ", step+1)
            
            self.shortest_paths.append(best_path)
            self.shortest_paths_lens.append(best_path_len)
            
            # Check for early stopping condition
            if self._early_stop:
                print("Early stopping condition met. Stopping ACO algorithm.")
                break
            
        output_index = self.shortest_paths_lens.index(min(self.shortest_paths_lens))
        output_path = self.shortest_paths[output_index]
        self.shortest_path_len = self.shortest_paths_lens[output_index]

        return output_path, self.shortest_paths, self.shortest_paths_lens

    #####
    
    def _calc_attraction(self):
        """
        Calculate the attractiveness of each state based on the distance between them (inversely proportional).
        """
        raise NotImplementedError("Attractiveness calculation is not implemented yet.")
        # for i, c in enumerate(self.states):
        #     for j, d in enumerate(self.states):
        #         distance = State.calc_distance(c, d)
        #         if distance > 0:
        #             self.attractiveness[i][j] = 1/distance
        #         else:
        #             self.attractiveness[i][j] = 0
    
    #####
    
    def _mp_tour(self, ant, q: Queue, multi_processing=True, display=False):
        """
        Perform a minimum-pheromone tour with the ant combined with DFS to ensure it reaches the goal.
        Parameters:
        - ant (Ant): The ant to perform the tour.
        - q (Queue): Queue to store the results.
        """
        parent = None  # Parent state for backtracking
        random.seed(os.getpid() + time.time())  # Set random seed for reproducibility per process 
        backtrack = False  # Flag to indicate if the ant is backtracking
        backtrack_counter = 0
        ant._init_ant(self)  # (re-)initialise the ant
        update_pheromone = True  # Flag to indicate if pheromone should be updated. TODO
        
        while(not ant.reached_goal):  # until the ant reaches the goal
            # if len(ant.path) < (self.height * self.width):
            #     print("Quitting tour because ant has become stuck...")
            #     update_pheromone = False
            #     break
            ant.set_reachable(self)  # determine all the possible moves based on adjacency and obstructions
            if display and not multi_processing:  # if display is enabled, show the maze with the ant
                self.maze.display(ant.pos, ant.possible_states, self.ax, clear_first=True)
            
            if not ant.possible_states:  # if no possible states, the ant is stuck                
                if len(ant.path) <= 1:
                    # print(f"Ant {ant.ant_id} is stuck and has no path to backtrack to. Exiting...")
                    update_pheromone = False
                    break
                # Pop back to previous state to retrace the path
                # print(f"Ant {ant.ant_id} stuck with no moves. Backtracking to previous position...")
                if parent is not None:
                    if parent == self.start:
                        # print(f"Ant {ant.ant_id} has reached the start position with nothing left to explore. Exiting...")
                        update_pheromone = False
                        break
                parent = ant.path[-2 - backtrack_counter]  # get the last unstuck state
                ant.step(parent)  # move to the last state
                backtrack = True
                backtrack_counter += 2
                continue
            
            else:
                backtrack = False
                backtrack_counter = 0
            
            if random.random() < self.epsilon:  # exploration. Randomly select a city
                # print("Randomly selecting a city")
                next_pos = ant.possible_states[random.randint(0, len(ant.possible_states) - 1)]
            
            else:  # exploitation. Select the next city based on pheromone levels
                for c in ant.possible_states:
                    ant.update_transition_prob(self, c, normalise=False)  # from current routing table                
                
                if np.sum(ant.transition_probs) == 0:  # if all transition probabilities are zero, select a random city
                    ant.transition_probs = np.ones_like(ant.transition_probs) / len(ant.transition_probs)
                else: 
                    ant.transition_probs = ant.transition_probs/sum(ant.transition_probs)  # manually normalise
                
                # Select the next city based on pheromone levels
                selection = np.random.choice(ant.possible_states, 1, p=ant.transition_probs)
                next_pos = selection[0]
            
            ant.step(next_pos)
        
        if multi_processing:  # if multiprocessing is enabled, put the ant in the queue
            assert q is not None, "Queue must be provided for multiprocessing"
            q.put(ant)
        
    #####
    
    def _remove_backtracks(self, a):
        """
        # Remove loops/backtracked segments from the path of the ant.
        This is done by checking if the current state is already in the cleaned path.
        Parameters:
        - a (Ant): The ant whose path needs to be updated.
        """
        cleaned_path = []
        detected = False  # Flag to indicate if a loop was detected
        for state in a.path:
            if state in cleaned_path:
                # Found a loop, backtrack by popping until we return to this state
                while cleaned_path and cleaned_path[-1] != state:
                    cleaned_path.pop()
                    detected = True
            else:
                cleaned_path.append(state)

        # Update the ant's path
        a.path = cleaned_path
        a.path_length = len(cleaned_path)
        # print("Found a loop. Cleaned it.")
        
    #####
    
    def _update_pheromone(self, a):
        """
        Update the pheromone levels based on the path taken by the ant. Deposited pheromone is inversely proportional to the path length.
        Parameters:
        - a (Ant): The ant that has completed its tour.
        """
        for k in range(0, len(a.path) - 1):
            try:
                i = a.path[k].index
                j = a.path[k + 1].index
                delta_pher = self.pheromone_deposit / (a.path_length**1)  # shorter paths get more pheromone

                self.pheromone[i][j] += delta_pher  # update pheromone level
                # self.pheromone[j][i] += delta_pher  # symmetric update not done as directional graph
                
            except IndexError as e:
                print(f"IndexError in path calculation: {e}")
                raise
    
    #####
    
    def _evaporate_pheromone(self):
        """
        Evaporate pheromone levels uniformly across the matrix -- typically done once per iteration.
        """
        self.pheromone *= (1 - self.evaporation_const)
        
    #####        
    
    def _update_routing_table(self, a):
        """
        Update the routing table based on the path taken by the ant.
        Parameters:
        - a (Ant): The ant that has completed its tour.
        """        
        for c in a.path:
            reachable = Ant.find_reachable(c, self)  # find all reachable states from the current state    
            for state in reachable:
                numer = Ant.calc_desirability(self, c, state)  # c <-> state
                denom = 0.00        
                for also_state in reachable:  # calculate normalisation factor
                    denom += Ant.calc_desirability(self, c, also_state)  # c <-> also_state (same as state)
                
                # Update routing table based on desirability
                if denom > 0:
                    self.routing_table[c.index][state.index] = numer/denom
                elif denom == 0:
                    print("Denominator is zero. Evenly distributing routing table probabilities.")
                    self.routing_table[c.index][state.index] = 1 / len(reachable)  # evenly distributed
                else:  # this should not happen
                    print("Denominator is negative...")
                    self.routing_table[c.index][state.index] = 0

    #####
    
    @property
    def _early_stop(self):
        """
        Check if the shortest path has been reached n times in a row. If yes, stop the algorithm.
        Parameters:
        - n (int): Number of times the shortest path must be repeated to trigger early stopping.
        """
        n = self.early_stopping_n
        if len(self.shortest_paths_lens) >= n:
            if all(self.shortest_paths_lens[-1] == self.shortest_paths_lens[-i] for i in range(2, n + 1)):
                print(f"Early stopping: Shortest path length repeated {n} times in a row.")
                return True
        return False
        
##################################################################

if __name__ == "__main__":
    # Example usage
    aco = ACO(num_states=100, initial_pheromone=1.0, alpha=1, beta=3, epsilon=0.1,
              pheromone_deposit=2, evaporation_constant=0.6, type_="maze", width=10, height=10)
    
    print(ACO)
    
##################################################################
