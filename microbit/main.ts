let width = 5;
let height = 5;
let start = [0, 0];
let goal = [height - 1, width - 1];
let initPher = 0.001; 
let epsilon = 0.1;
let scanBools = [false, false, false, false];
let allowedMoves = [0, 90, 180, 270];
let orientation = 0;
let numReverses = 0;
let nStates = width * height;
let phArray: number[] = []; 
for (let i = 0; i < nStates * nStates; i++) {
    phArray.push(initPher);
}

input.onButtonPressed(Button.A, function () {
    phArray = solveMaze(start, goal, orientation, epsilon, phArray);
})

function scanEnvironment(): boolean[] {
    let scanBools = [false, false, false, false];
    let retryCounter = 0;
    while(retryCounter < 5) {
        //  Reset scan_bools
        CutebotPro.trackbitStateValue();
        if (CutebotPro.getGrayscaleSensorState(TrackbitStateType.Tracking_State_1)) {
            //  Straight path
            scanBools[0] = true;
            return scanBools;
        } else if (CutebotPro.getGrayscaleSensorState(TrackbitStateType.Tracking_State_9)) {
            //  Left turn
            scanBools[2] = true;
            return scanBools;
        } else if (CutebotPro.getGrayscaleSensorState(TrackbitStateType.Tracking_State_13)) {
            //  Right turn
            scanBools[3] = true;
            return scanBools;
        } else {
            //  Reverse / No forward path / Anything else
            //  (CutebotPro.getGrayscaleSensorState(TrackbitStateType.Tracking_State_0))
            retryCounter ++;
            basic.pause(1000);
        }
    }
    scanBools[1] = true
    return scanBools;
}

function moveCutebot(newMove: number, orientation: number, newOrientation: any = null): number {
    if (newMove == 0) {
        orientation = newOrientation !== null ? newOrientation : orientation;
        CutebotPro.distanceRunning(CutebotProOrientation.Advance, 5, CutebotProDistanceUnits.Cm);
    } else if (newMove == 180) {
        CutebotPro.distanceRunning(CutebotProOrientation.Retreat, 5, CutebotProDistanceUnits.Cm);
        //  orientation = (orientation + 180) % 360  # Update orientation after reverse
        orientation = newOrientation !== null ? newOrientation : (orientation + 180) % 360;
    } else if (newMove == 90) {
        //  Update after reverse
        CutebotPro.angleRunning(CutebotProWheel.LeftWheel, -45, CutebotProAngleUnits.Angle);
        CutebotPro.angleRunning(CutebotProWheel.RightWheel, 45, CutebotProAngleUnits.Angle);
        CutebotPro.distanceRunning(CutebotProOrientation.Advance, 5, CutebotProDistanceUnits.Cm);
        orientation = newOrientation !== null ? newOrientation : (orientation + 90) % 360;
    } else if (newMove == 270) {
        //  Update after left turn
        CutebotPro.angleRunning(CutebotProWheel.LeftWheel, 45, CutebotProAngleUnits.Angle);
        CutebotPro.angleRunning(CutebotProWheel.RightWheel, -45, CutebotProAngleUnits.Angle);
        CutebotPro.distanceRunning(CutebotProOrientation.Advance, 5, CutebotProDistanceUnits.Cm);
        orientation = newOrientation !== null ? newOrientation : (orientation - 90) % 360;
    } else {
        //  Update after right turn
        control.fail("Unrecognized move!");
    }

    return orientation
}

function solveMaze(
    start: number[],
    goal: number[],
    orientation: number,
    epsilon: number,
    phArray: number[]): number[] {
    
    let pos = start.slice();
    let path: number[] = [ACO.xyToState(pos[0], pos[1])];
    let movesTaken: number[] = [];    
    let reverseCounter = 0; 
    let pathLength = 1;                 
    ACO.showOrientation(0);

    while (pos[0] !== goal[0] || pos[1] !== goal[1]) {
        const scanBools = scanEnvironment();
        
        if (scanBools[1] && pathLength == 1){
            basic.showString("PA2R!");
            return []
        }
        
        const viableMoves = ACO.getViableMoves(scanBools, pos, path, orientation);

        if (viableMoves.length > 0) {
            const newMove = ACO.selectMove(viableMoves, epsilon, phArray, pos, orientation);
            const newPos = ACO.step(newMove, orientation, pos);

            orientation = moveCutebot(newMove, orientation);
            ACO.showOrientation(orientation);

            path.push(ACO.xyToState(newPos[0], newPos[1]));
            movesTaken.push(newMove);
            pathLength++;

            let localDeposit = (newPos[0] === goal[0] && newPos[1] === goal[1]) ? 10 : 1;
            phArray = ACO.dropPheromone(pos, newPos, pathLength, phArray, localDeposit);

            pos = newPos.slice();
            reverseCounter = 0;

        } else {
            let backIdx = path.length - 2 - reverseCounter;
            if (backIdx < 0) break;

            const flatState = path[backIdx];
            pos = ACO.stateToXy(flatState);

            const [newOrientation, newCounter] = ACO.reverseLastMove(movesTaken, reverseCounter, orientation);
            reverseCounter = newCounter;

            orientation = moveCutebot(180, orientation, newOrientation);

            path.push(flatState);
            movesTaken.push(180);
        }
    }
    return phArray
}
