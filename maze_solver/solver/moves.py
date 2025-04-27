from enum import Enum

##################################################################

class Moves(Enum):
    # Enumerate has two properties by default: name and value
    # Using standard python directions for array indexing
    UP =    ( -1,  0 )   # (dx, dy): Up decreases x
    DOWN =  ( +1,  0 )   # Down increases x
    LEFT =  (  0, -1 )   # Left decreases y
    RIGHT = (  0, +1 )   # Right increases y

    @property
    def dx(self):
        return self.value[0]  # Access x-offset

    @property
    def dy(self):
        return self.value[1]  # Access y-offset

    def __str__(self):
        return self.name
    
##################################################################

if __name__ == "__main__":
    # Example usage
    
    move = Moves.RIGHT
    print(move.value)  # Output: (1, 0)

    # Access dx/dy individually
    print(move.dx)     # Output: 1 (x-offset)
    print(move.dy)     # Output: 0 (y-offset)
    
    # Iterate through all moves
    for move in Moves:
        print(f"{move}: dx={move.dx}, dy={move.dy}")
    
##################################################################
