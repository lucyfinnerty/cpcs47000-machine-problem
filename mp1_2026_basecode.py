#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Name: Christopher Lodzinski, Lucille Finnerty, Jeffrey Elsenbach, and Bret Evenson
Date: 2026-09-27
Course: Artificial Intelligence
Semester: Fall 2026
Assignment: MP1 - Robot navigation

"""

import numpy as np
import queue
from heapq import heapify


class MazeState():
    """ Stores information about each visited state within the search """
    # Define constants
    SPACE = 0
    WALL = 1
    EXIT = 2
    VISITED = 3
    PATH = 4
    START_MARK = 5
    END_MARK = 6

    MAZE_FILE = 'maze2026.txt'
    maze = np.loadtxt(MAZE_FILE, dtype=np.int32)  
    start = tuple(np.array(np.where(maze==5)).flatten())
    ends = np.where(maze==2)
    move_num = 0 # Used by show_path() to count moves in the solution path
    
    def reset_state():
        """ Resets the static variables to prepare for a new search """
        MazeState.maze = np.loadtxt(MazeState.MAZE_FILE, dtype=np.int32)  
        MazeState.start = tuple(np.array(np.where(MazeState.maze==5)).flatten())
        MazeState.ends = np.where(MazeState.maze==2)
        MazeState.move_num = 0
    
    def __init__(self, conf=None, g=0, pred_state=None, pred_action=None):
        """ Initializes the state with information passed from the arguments """
        self.pos = conf if conf is not None else MazeState.start  # Configuration of the state -current coordinates
        self.gcost = g          # Path cost
        self.pred = pred_state  # Predecesor state
        self.action_from_pred = pred_action  # Action from predecesor state to current state
    
    def __hash__(self):
        """ Returns a hash code so that it can be stored in a set data structure """
        return self.pos.__hash__()
    
    def is_goal(self):
        """ Returns true if current position is same as the exit position """
        return self.maze[self.pos] == MazeState.EXIT
    
    def __eq__(self, other):
        """ Checks for equality of states by positions only """
        return self.pos == other.pos
    
    def __lt__(self, other):
        """ Allows for ordering the states by the path (g) cost """
        return self.fcost < other.fcost
    
    def __str__(self):
        """ Returns the maze representation of the state """
        a = np.array(self.maze)
        a[self.start] = MazeState.START_MARK
        a[self.ends] = MazeState.EXIT
        return str(a)

    def show_path(self):
        """ Recursively outputs the list of moves and states along path """
        if self.pred is not None:
            self.pred.show_path()
        
        if MazeState.move_num==0:
            print('START')
        else:
            print('Move',MazeState.move_num, 'ACTION:', self.action_from_pred)
        MazeState.move_num = MazeState.move_num + 1
        self.maze[self.pos] = MazeState.PATH
    
    def get_new_pos(self, move):
        """ 
        Returns a new position from the current position and the specified move. 
        Modulo operations implement wrap-around movement 
        """
        num_rows, num_cols = self.maze.shape
        if move=='up':
            new_pos = ((self.pos[0]-1) % num_rows, self.pos[1])
        elif move=='down':
            new_pos = ((self.pos[0]+1)% num_rows, self.pos[1])
        elif move=='left':
            new_pos = (self.pos[0], (self.pos[1]-1) % num_cols)
        elif move=='right':
            new_pos = (self.pos[0], (self.pos[1]+1) % num_cols)
        else:
            raise ValueError('wrong direction for checking move')
        return new_pos
        
    def can_move(self, move, disabled_move=None):
        """ Returns true if agent can move in the given direction """
        if move == disabled_move:
            return False
        new_pos = self.get_new_pos(move)
         # the move is allowed if the destination is not a wall
        return self.maze[new_pos]!=MazeState.WALL

    def heuristic(self):
        """ Returns the heuristic value for the current state """
        num_rows, num_cols = self.maze.shape
        minDist = float('inf')

        # check the distance to each exit and return the minimum distance
        for i in range(len(self.ends[0])):
            exit_row = self.ends[0][i]
            exit_col = self.ends[1][i]

            row_distance = abs(self.pos[0] - exit_row)
            col_distance = abs(self.pos[1] - exit_col)

            # account for wrap-around movement
            row_distance = min(row_distance, num_rows - row_distance)
            col_distance = min(col_distance, num_cols - col_distance)

            total_distance = row_distance + col_distance
            
            if total_distance < minDist:
                minDist = total_distance

        return minDist
    
    def gen_next_state(self, move):
        """  
        Generates a new MazeState object by taking move from current
        state. The f-cost is calculated as g-cost + heuristic.
        """
        new_pos = self.get_new_pos(move)
        if self.maze[new_pos] != MazeState.EXIT:
            self.maze[new_pos] = MazeState.VISITED
        new_state = MazeState(new_pos, self.gcost+1, self, move)
        new_state.fcost = new_state.gcost + new_state.heuristic()

        return new_state
    
    def run_astar(disabled_move):
        """ 
        Runs the A* search algorithm to find the shortest path to the exit while
        disabling the specified movement direction.

        A* selects states using f(n) = g(n) + h(n), where g(n)
        is the path cost so far and h(n) is the heuristic estimate
        of the remaining distance to the closest exit.
        """
        # reset the maze for a new run
        MazeState.reset_state()
    
        # load start state onto frontier priority queue
        frontier = queue.PriorityQueue()
        start_state = MazeState()
        start_state.fcost = start_state.gcost + start_state.heuristic()
        frontier.put(start_state)
    
        # keep a closed set of states to which optimal path was already found
        closed_set = set()
    
        # expand state (up to 4 moves possible)
        possible_moves = ['left', 'right', 'down', 'up']
    
        num_states = 0
        goal_state = None # initialize the goal state

        while not frontier.empty():
            # choose state at front of priority queue
            next_state = frontier.get()
            num_states += 1
    
            # if goal then quit and return path
            if next_state.is_goal():
                goal_state = next_state
                break
    
            # add state chosen for expansion to closed_set
            closed_set.add(next_state)
    
            # expand the current state by generating all valid moves
            for move in possible_moves:
                # added disabled_move to skip the disabled direction
                if next_state.can_move(move, disabled_move):
                    neighbor = next_state.gen_next_state(move)
                    if neighbor in closed_set:
                        continue
                    if neighbor not in frontier.queue:
                        frontier.put(neighbor)
                    else:
                        # find and compare old route to new route
                        index = frontier.queue.index(neighbor)
                        # keep it if it is a better route
                        if neighbor.fcost < frontier.queue[index].fcost:
                            frontier.queue[index] = neighbor
                            # resort the list
                            heapify(frontier.queue)
    
        # added a case for if there is no solution to the maze
        if goal_state is None:
            print(start_state)
            print('No solution')
            return None

        # display solution path
        goal_state.show_path()
        print(start_state)
        print('\nNumber of states visited =', num_states)
        path_length = MazeState.move_num - 1
        print('\nLength of shortest path = ', path_length)
        return path_length

# display the heading info
print('Artificial Intelligence')
print('MP1: Robot navigation')
print('SEMESTER: Fall 2026')
print('NAME: Christopher Lodzinski, Lucille Finnerty, Jeffrey Elsenbach, and Bret Evenson')
print()

print('INITIAL MAZE')
print(MazeState())

# try each move as the disabled one and keep the shortest solution
best_move = None
best_length = None
for move in ['left', 'right', 'down', 'up']:
    print('SOLUTION AFTER DISABLED MOVE: ', move)
    length = MazeState.run_astar(move)
    if length is not None and (best_length is None or length < best_length):
        best_move = move
        best_length = length

print('BEST MOVE: disable', best_move)
print('SHORTEST PATH LENGTH FOR BEST MOVE:', best_length)