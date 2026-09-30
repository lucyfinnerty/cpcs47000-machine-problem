#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Name: [your name]
Date: 2026-09-27
Course: Artificial Intelligence
Semester: Fall 2026
Assignment: MP1 - Robot navigation

"""

import numpy as np
import queue # Needed for frontier queue (wont need for new implementation)
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

    MAZE_FILE = 'maze2024.txt'
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
    
    def __init__(self, conf=start, g=0, pred_state=None, pred_action=None):
        """ Initializes the state with information passed from the arguments """
        self.pos = conf         # Configuration of the state - current coordinates
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
        #return self.gcost < other.gcost
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
        """ Returns a new position from the current position and the specified move 
        *UPDATED: the modulo operations implement wrap-around movement 
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
            raise('wrong direction for checking move')
        return new_pos
        
    """
    *UPDATED: The can_move() method now implements wrap-around movement. 
    The commented-out code is the original version that did not allow wrap-around 
    movement.
    """
    def can_move(self, move, disabled_move=None):
        """ Returns true if agent can move in the given direction """
        if move == disabled_move:
            return False
        new_pos = self.get_new_pos(move)

        #if new_pos[0] < 0 or new_pos[0] >= self.maze.shape[0] or new_pos[1] < 0 or new_pos[1] >= self.maze.shape[1]:
        #    return False
        #else:

        return self.maze[new_pos]!=MazeState.WALL



    # -----------------IMPLEMENT CODE-------------------
    #def heuristic(self):
        """ Returns the heuristic value for the current state """

    
    def gen_next_state(self, move):
        """ Generates a new MazeState object by taking move from current state 
        *UPDATED: The new state is now generated with the fcost value calculated as
        fcost = gcost + heuristic()
        """
        new_pos = self.get_new_pos(move)
        if self.maze[new_pos] != MazeState.EXIT:
            self.maze[new_pos] = MazeState.VISITED
        #return MazeState(new_pos, self.gcost+1, self, move)
        new_state = MazeState(new_pos, self.gcost+1, self, move)
        new_state.fcost = new_state.gcost + new_state.heuristic()

        return new_state



    # -----------------IMPLEMENT CODE-------------------
    #def run_astar(disabled_move):
        """ Runs the A* search algorithm to find the shortest path to the exit """


#-----------EDIT BELOW THIS LINE FOR NEW A* IMPLEMENTATION----------------

# Display the heading info
print('Artificial Intelligence')
print('MP1: Robot navigation')
print('SEMESTER: Fall 2026')
print('NAME: [your name]')
print()

print('INITIAL MAZE')

# load start state onto frontier priority queue
frontier = queue.PriorityQueue() # This does best-first search
#frontier = queue.LifoQueue() # This would do depth-first search
#frontier = queue.Queue() # This would do breadth-first search

start_state = MazeState()
frontier.put(start_state)
print(start_state)
# Keep a closed set of states to which optimal path was already found
closed_set = set()

# Expand state (up to 4 moves possible)
possible_moves = ['left','right','down','up']

num_states = 0
while not frontier.empty():
    # Choose state at front of priority queue
    next_state = frontier.get()
    num_states = num_states + 1
    
    # If goal then quit and return path
    if next_state.is_goal():
        next_state.show_path()
        break
    
    # Add state chosen for expansion to closed_set
    closed_set.add(next_state)
  
    # Expanding the node
    for move in possible_moves:
        if next_state.can_move(move):
            neighbor = next_state.gen_next_state(move)
            if neighbor in closed_set:
                continue
            if neighbor not in frontier.queue:                           
                frontier.put(neighbor)
            else:
                if neighbor.gcost < frontier.queue[frontier.queue.index(neighbor)].gcost:
                    frontier.queue[frontier.queue.index(neighbor)] = neighbor
                    heapify(frontier.queue)

print(start_state)                
print('\nNumber of states visited =',num_states)
move_path_length = MazeState.move_num-1
print('\nLength of shortest path = ', move_path_length)