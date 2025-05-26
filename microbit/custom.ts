// custom.ts

// enum moves {
//     //% block="forward"
//     forward,
//     //% block="backward"
//     backward,
//     //% block="left"
//     left,
//     //% block="right"
//     right,
//     //% block="stop"
//     nill,
// }

namespace ACO {
    let viable_moves: number[] = []; // Initialize
    let width = 5, height = 5, nStates = 25
    let deltas = []
    let new_idx: number = 0
    let new_pos: number[] = []
    let pher_level: number[] = []
    let last_move: number = 0

    /*********************************************************************/
    /**
    * Convert a 2-D cell coordinate (row, col) into a flat state index 0…nStates-1
    */
    function xyToState(row: number, col: number): number {
        return row * width + col;
    }

    /**
     * Given two state-indices i,j, return the index into your
     * flattened nStates×nStates pheromone array.
     * */
    function idx(i: number, j: number): number {
        return i * nStates + j;
    }
    /**
        * Determine the viable moves based on the scan outputs.
        * @param scan_outputs an array of booleans representing the state of each sensor.
        * @param pos current index of the ant on the maze.
        * @param path an array of numbers representing the path taken so far.
        * @orientation the direction in which the ant is pointing. Convention is positive coordinate system with East = 0 deg
        * The path is used to avoid revisiting already traversed states.
        * @returns a list of "moves" representing the viable next move.
     */
    //% blockId="ACO_get_viable_moves"
    //% block="get_viable_moves "
    export function get_viable_moves(scan_outputs: boolean[], pos: number[], path: number[], 
                                     orientation: number): number[] {
        // For now, we return a placeholder value.
        if (scan_bools[0]){
            new_pos = step(0, orientation, pos);
            new_idx = idx(new_pos[0], new_pos[1])
            if (path.indexOf(new_idx) === -1) {
                viable_moves.push(0);
            }
        }
        else{
            if (scan_bools[1]) {
                new_pos = step(180, orientation, pos);
                new_idx = idx(new_pos[0], new_pos[1])
                if (path.indexOf(new_idx) === -1) {
                    viable_moves.push(180);
                }
            }
            else{
                if (scan_bools[2]) {
                    new_pos = step(90, orientation, pos);
                    new_idx = idx(new_pos[0], new_pos[1])
                    if (path.indexOf(new_idx) === -1) {
                        viable_moves.push(90);
                    }
                }    
                else{
                    if (scan_bools[3]) {
                        new_pos = step(270, orientation, pos);
                        new_idx = idx(new_pos[0], new_pos[1])
                        if (path.indexOf(new_idx) === -1) {
                            viable_moves.push(270);
                        }
                    }
                }
            }
        }
        return viable_moves;
    }

    /*****/

    /**
        * Determine the next move based on the available moves and the pheromone levels.
        * @param viable_moves an list of enum of numbers representing the available moves.
        * @param epsilon a number representing the exploration rate.
        * @param ph_array an array of numbers representing the pheromone levels for each move (nStates x nStates)
        * @param pos current index of the ant on the maze.
        * @orientation the direction in which the ant is pointing. Convention is positive coordinate system with East = 0 degrees
        * @returns an array of two numbers representing the next move in the format [x, y]
     */
    //% blockId="ACO_select_move"
    //% block="select_move "
    export function select_move(viable_moves: number[], epsilon: number, ph_array: number[], 
                                pos: number[], orientation: number): number {
        const n = viable_moves.length;
        
        // only one choice
        if (n === 1) return viable_moves[0];
        // 1) Exploration
        if (Math.random() < epsilon) {
            // uniform random from viable_moves
            const ri = Math.randomRange(0, n - 1);
            return viable_moves[ri];
        }
        // 2) Exploitation
        const fromState = xyToState(pos[0], pos[1]);
        const weights: number[] = [];
        for (let mv of viable_moves) {
            const new_pos = step(mv, orientation, pos);
            const toState = xyToState(new_pos[0], new_pos[1]);
            const edgeIdx = idx(fromState, toState);
            weights.push(ph_array[edgeIdx]);
        }

        // sum of all pheromones
        const total = weights.reduce((sum, w) => sum + w, 0);

        // normalize into probabilities
        const probs = weights.map(w => w / total);

        // weighted random choice
        return weightedRandomChoice(viable_moves, probs);

        /**
         * Pick one item from `items` with given (same‐length) `probs` array.
         * @returns one element of `items`
         */
        function weightedRandomChoice<T>(items: T[], probs: number[]): T {
            let r = Math.random();
            let cumulative = 0;
            for (let i = 0; i < items.length; i++) {
                cumulative += probs[i];
                if (r < cumulative) return items[i];
            }
            // safety
            return items[items.length - 1];
        }
    }

    /*****/

    /**
        * Reverses last move: adjusts the orientation
        * @param moves_taken the moves the robot has made so far
        * @param reverse_counter counter which tracks the number of reverse the robot makes (backtracking)
        * @orientation the direction in which the ant is pointing. Convention is positive coordinate system with East = 0 degrees
        * @returns an array of two numbers representing the delta x
     */
    //% blockId="ACO_reverse_last_move"
    //% block="reverse_last_move "
    export function reverse_last_move(moves_taken: number[], reverse_counter: number, orientation: number): [number, number] {
        
        // Initialisations
        const idx = moves_taken.length - 1 - reverse_counter;
        const last_move = moves_taken[idx];
        let new_orientation = orientation;

        // Undo the last move
        if(last_move == 90) new_orientation = (orientation + 270) % 360;
        else if(last_move == 270) new_orientation = (orientation + 90) % 360;
        
        // Increment reverse counter
        reverse_counter += 2;

        return [new_orientation, reverse_counter]
    }

    /*****/

    /**
        * Calculates next position based on current position, new move and the orientation
        * @param move the move to translate
        * @param orientation the current orientation of the robot in degrees (0 = right, 90 = up, 180 = left, 270 = down).
        * @returns an array of two numbers representing the delta x and y values.
     */
    //% blockId="ACO_step"
    //% block="step "
    export function step(move: number, orientation: number, pos: number[]): number[] {
        deltas = moveToDelta(move, orientation);
        let delta_x = deltas[0];
        let delta_y = deltas[1];
        let new_pos = [pos[0] + delta_x, pos[1] + delta_y];
        return new_pos;
    }

    /*****/

    /**
        * Translate moves enum to delta x and y values. 
        * Convention is (forward/backward, left/right). Assuming the robot is pointing right. 
        * @param move to make.
        * @param orientation the current orientation of the robot in degrees (0 = E, 90 = N, 180 = W, 270 = S).
        * @returns an array of two numbers representing the delta x and y values. 
     */
    //% blockId="ACO_move_to_delta"
    //% block="move_to_delta "
    export function moveToDelta(move: number, orientation: number): number[] {
        switch (move) {
            case 0:
                switch (orientation) {
                    case 0: return [0, 1];    // right
                    case 90: return [-1, 0];   // up
                    case 180: return [0, -1]; // left
                    case 270: return [1, 0]; // down
                }
                break;
            case 180:
                switch (orientation) {
                    case 0: return [0, -1];   // right
                    case 90: return [1, 0];  // up
                    case 180: return [0, 1];   // left
                    case 270: return [-1, 0];   // down
                }
                break;
            case 90:
                switch (orientation) {
                    case 0: return [-1, 0];   // right
                    case 90: return [0, -1];   // up
                    case 180: return [1, 0];   // left
                    case 270: return [0, 1];   // down
                }
                break;
            case 270:
                switch (orientation) {
                    case 0: return [1, 0];    // right
                    case 90: return [0, 1];   // up
                    case 180: return [-1, 0]; // left
                    case 270: return [0, -1]; // down
                }
                break;
            default:
                return [0, 0]; // No movement
        }
        return [0, 0]; // Default case, no movement
    }

    /*****/
    /**
        * Drops the pheromone along the trail. The ant does this as it goes.
        * @param prev_pos previous position of the ant in the maze
        * @param new_pos new position of the ant in the maze
        * @param path_length length of the path taken so far
        * @param ph_array pheromone array to update
        * @param deposit_amount amount of pheromone to deposit_amount
        * @returns the updated pheromone array
     */
    //% blockId="ACO_drop_pheromone"
    //% block="drop_pheromone "
    export function drop_pheromone(prev_pos: number[], new_pos: number[], path_length: number, ph_array: number[], deposit_amount: number): number[] {
        // Map coordinates into grid indices
        const fromState = xyToState(prev_pos[0], prev_pos[1]);
        const toState = xyToState(new_pos[0], new_pos[1]);

        // Compute the edge index for this transition
        const edgeIdx = idx(fromState, toState);

        // Compute how much pheromones to drop
        const delta_pher = deposit_amount / path_length;

        // Deposit the pheromone
        ph_array[edgeIdx] += delta_pher; 

        return ph_array;
    }

    /*****/
    /**
        * Shows the current orientation of the ant on the screen.
        * @param orientation the current orientation of the robot in degrees (0 = E, 90 = N, 180 = W, 270 = S).
     */
    //% blockId="ACO_show_orientation"
    //% block="show_orientation "

    export function show_orientation(orientation: number): number {
        switch (orientation) {
            case 0: basic.showArrow(ArrowNames.East);
            case 90: basic.showArrow(ArrowNames.North);
            case 180: basic.showArrow(ArrowNames.West);
            case 270: basic.showArrow(ArrowNames.South);
        }
        return 0
    }
}