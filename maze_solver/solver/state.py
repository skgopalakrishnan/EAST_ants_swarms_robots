from typing import Optional
import numpy as np

##################################################################

class State:
    def __init__(self, x: int, y: int, index: Optional[int] = None):
        self.x = x
        self.y = y
        self.index = index  # assumed to be in row-major order

    #####
    
    def __repr__(self):
        if self.index is not None:
            return f"State:({self.x}, {self.y}, index={self.index})"
        return f"State:({self.x}, {self.y})"

    #####

    def __eq__(self, other):
        return (self.x, self.y, self.index) == (other.x, other.y, other.index)

    #####

    def __hash__(self):
        return hash((self.x, self.y, self.index))
    
    #####
    
    def __iter__(self):
        return iter((self.x, self.y))

    #####

    @staticmethod
    def calc_distance(state1, state2):
        """
        Calculate the distance between two states.
        Parameters:
            state1 (State): First state.
            state2 (State): Second state.
        Returns:
            float: Distance between the two states.
        """
        assert isinstance(state1, State), "state1 must be an instance of State"
        assert isinstance(state2, State), "state2 must be an instance of State"
        
        # Calculate Euclidean distance between state1 and state2
        return np.sqrt((state1.x - state2.x)**2 + (state1.y - state2.y)**2)
    
    #####
    
    @staticmethod
    def coords_to_idx(x: int, y: int, grid_dimensions: tuple) -> int:
        """
        Convert (x, y) coordinates to a 1D index based on row-major order.
        """
        return x * grid_dimensions[1] + y
    
    #####
    
    @staticmethod
    def idx_to_coords(index: int, grid_dimensions: tuple) -> tuple:
        """
        Convert a 1D index to (x, y) coordinates based on row-major order.
        """
        return divmod(index, grid_dimensions[1])

##################################################################

if __name__ == "__main__":
    # Example usage
    state = State(1, 2, index=3)
    print(state)  # Output: State:(1, 2, index=3)
    
    state2 = State(1, 2)
    print(state == state2)  # Output: False as index is different
    print(tuple(state))  # Output: (1, 2)
    
    # Convert coordinates to index
    grid_dimensions = (5, 5)  # Example grid dimensions
    x, y = -1, -1
    index = State.coords_to_idx(x, y, grid_dimensions)
    
    print(f"Index of ({x}, {y}) in grid {grid_dimensions}: {index}")  # Output: 13
    
    # Convert index to coordinates
    index = 13
    coords = State.idx_to_coords(index, grid_dimensions)
    print(f"Coordinates of index {index} in grid {grid_dimensions}: {coords}")  # Output: (2, 3)
    
##################################################################
