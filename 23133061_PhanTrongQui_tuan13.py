import pygame
import time
import heapq
import random
import math
from collections import deque
import sys
import traceback
import textwrap
import threading
import queue
import os # Needed for font check

# --- Pygame Initialization and Basic Setup ---
try:
    pygame.init()
    # Check if font module initialized correctly
    if not pygame.font:
        print("Pygame Font module not initialized!")
        # Attempt to re-initialize specifically
        pygame.font.init()
        if not pygame.font:
             raise RuntimeError("Pygame Font module failed to initialize.")
    print("Pygame initialized successfully.")
except Exception as e:
    print(f"Fatal Error initializing Pygame: {e}")
    sys.exit(1)

# --- Screen Dimensions and Layout ---
GRID_SIZE = 450         # Pixel size of the puzzle grid area
TILE_SIZE = GRID_SIZE // 3
PANEL_PADDING = 15      # Padding between major UI sections
CONTROLS_WIDTH = 220    # Width for buttons and sliders
RESULTS_WIDTH = 500     # Width for the results/log area
BOTTOM_MARGIN = 60      # Space at the bottom for sliders/notes (Increased slightly)

WIDTH = GRID_SIZE + CONTROLS_WIDTH + RESULTS_WIDTH + PANEL_PADDING * 4
HEIGHT = GRID_SIZE + PANEL_PADDING * 2 + BOTTOM_MARGIN

GRID_X = PANEL_PADDING
GRID_Y = PANEL_PADDING
CONTROLS_X = GRID_X + GRID_SIZE + PANEL_PADDING
CONTROLS_Y = GRID_Y
RESULTS_X = CONTROLS_X + CONTROLS_WIDTH + PANEL_PADDING
RESULTS_Y = GRID_Y
RESULTS_HEIGHT = GRID_SIZE # Initial results height, matches grid

# Area below controls/results for sliders and notes
BOTTOM_PANEL_Y = GRID_Y + GRID_SIZE + PANEL_PADDING
BOTTOM_PANEL_HEIGHT = BOTTOM_MARGIN
BOTTOM_PANEL_WIDTH = CONTROLS_WIDTH + RESULTS_WIDTH + PANEL_PADDING # Span both columns
BOTTOM_PANEL_X = CONTROLS_X

# --- Colors (Pygame uses RGB tuples) ---
WHITE = (255, 255, 255); BLACK = (0, 0, 0); GREY = (200, 200, 200)
DARK_GREY = (100, 100, 100); LIGHT_GREY = (240, 240, 240) # Background
RED = (200, 0, 0); GREEN = (0, 200, 0); BLUE = (70, 130, 180) # SteelBlue
DARK_BLUE = (0, 0, 139); ORANGE = (255, 165, 0); PURPLE = (128, 0, 128)
DARK_RED = (150, 50, 50); DARK_GREEN = (0, 100, 0)
NOTE_COLOR = (80, 80, 80)
TILE_BORDER = (64, 64, 64)
TILE_EMPTY_BG = (220, 220, 220) # Gainsboro
BUTTON_COLOR = (0, 120, 215) # Nice blue
BUTTON_HOVER_COLOR = (0, 100, 185)
BUTTON_DISABLED_COLOR = (160, 160, 160)
BUTTON_TEXT_COLOR = WHITE
BUTTON_DISABLED_TEXT_COLOR = DARK_GREY
SLIDER_TRACK_COLOR = GREY
SLIDER_KNOB_COLOR = BUTTON_COLOR
SLIDER_KNOB_BORDER = DARK_GREY
TEXT_COLOR = BLACK

MSG_DEFAULT_COLOR = BLACK
MSG_SUCCESS_COLOR = DARK_GREEN
MSG_ERROR_COLOR = RED
MSG_INFO_COLOR = BLUE
MSG_WARN_COLOR = ORANGE

TILE_COLORS = [
    (255, 99, 71), (255, 127, 80), (255, 140, 0), (255, 160, 122), # Reds/Oranges
    (144, 238, 144), (143, 188, 143), (60, 179, 113), (46, 139, 87), # Greens
    (173, 216, 230) # Light Blue for 8
]

# --- Fonts ---
try:
    FONT_PATH = pygame.font.match_font(['segoeui', 'calibri', 'arial', 'sans'])
    if FONT_PATH is None: FONT_PATH = pygame.font.get_default_font()
    print(f"Using font: {FONT_PATH}")

    TILE_FONT = pygame.font.Font(FONT_PATH, 50)
    BUTTON_FONT = pygame.font.Font(FONT_PATH, 16)
    MSG_FONT = pygame.font.Font(FONT_PATH, 17)
    # Use a common monospace font if available
    MONO_FONT_PATH = pygame.font.match_font(['consolas', 'couriernew', 'mono'])
    if MONO_FONT_PATH is None: MONO_FONT_PATH = pygame.font.get_default_font()
    RESULTS_FONT = pygame.font.Font(MONO_FONT_PATH, 14)
    NOTE_FONT = pygame.font.Font(FONT_PATH, 14)
    SLIDER_FONT = pygame.font.Font(FONT_PATH, 13)
except Exception as e:
    print(f"Error loading fonts: {e}. Using default font.")
    DEFAULT_FONT = pygame.font.get_default_font()
    TILE_FONT = pygame.font.Font(DEFAULT_FONT, 50)
    BUTTON_FONT = pygame.font.Font(DEFAULT_FONT, 16)
    MSG_FONT = pygame.font.Font(DEFAULT_FONT, 17)
    RESULTS_FONT = pygame.font.Font(DEFAULT_FONT, 14)
    NOTE_FONT = pygame.font.Font(DEFAULT_FONT, 14)
    SLIDER_FONT = pygame.font.Font(DEFAULT_FONT, 13)

# --- Trạng thái đích và Giới hạn (Define constants FIRST) ---
GOAL_STATE = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
GOAL_STATE_TUPLE = tuple(map(tuple, GOAL_STATE)) # Now defined
DEFAULT_MAX_DEPTH = 15; MAX_BFS_STATES = 500000; MAX_DFS_POPS = 750000
MAX_GREEDY_EXPANSIONS = 500000; MAX_HILL_CLIMBING_STEPS = 5000
MAX_SA_ITERATIONS = 30000; MAX_BEAM_STEPS = 1000; MAX_GA_GENERATIONS = 300
MAX_IDA_STAR_NODES = 2000000

# --- Helper Functions (Corrected Order and Logic) ---
def state_to_tuple(state):
    """Converts a list-of-lists state to a tuple-of-tuples."""
    return tuple(map(tuple, state))

def is_valid_state(state):
    """Checks if the state is a valid 3x3 grid with numbers 0-8."""
    try:
        # Handle if input is already tuple of tuples or list of lists
        if isinstance(state, (list, tuple)) and len(state) == 3 and all(len(row) == 3 for row in state):
             # Flatten based on type
             if isinstance(state[0], tuple):
                 flat = sum(state, ()) # Use empty tuple for summing tuples
             else:
                 flat = sum(state, []) # Use empty list for summing lists
             return len(flat) == 9 and sorted(flat) == list(range(9))
        else: return False # Not a 3x3 list/tuple structure
    except (TypeError, ValueError):
        return False

def get_inversions(flat_state):
    """Calculates the number of inversions in a flattened state list."""
    count = 0; size = len(flat_state)
    for i in range(size):
        for j in range(i + 1, size):
            # Ensure elements are comparable integers and not the blank
            val_i, val_j = flat_state[i], flat_state[j]
            if isinstance(val_i, int) and isinstance(val_j, int) and \
               val_i != 0 and val_j != 0 and val_i > val_j:
                count += 1
    return count

def is_solvable(state, goal_state_ref=GOAL_STATE_TUPLE):
    """Checks if a state is solvable with respect to the standard goal."""
    if not is_valid_state(state):
        # print(f"State {state} is invalid.") # Debug
        return False

    # Standard solvability check compares against the standard 0-8 goal's inversion parity.
    standard_goal_flat = list(range(9))
    target_inversions_parity = get_inversions(standard_goal_flat) % 2

    # Flatten the input state correctly (list or tuple)
    if isinstance(state, tuple):
        state_flat = sum(state, ())
    else:
        state_flat = sum(state, [])

    state_inversions = get_inversions(state_flat)
    # print(f"State: {state}, Inversions: {state_inversions}, Target Parity: {target_inversions_parity}") # Debug
    return state_inversions % 2 == target_inversions_parity

def generate_random_solvable_state(goal_state_ref=GOAL_STATE_TUPLE):
    """Generates a random, solvable 3x3 state."""
    attempts = 0; max_attempts = 1000
    # Solvability is checked against the standard goal's parity.
    standard_goal_flat = list(range(9))
    target_inversions_parity = get_inversions(standard_goal_flat) % 2

    while attempts < max_attempts:
        flat = list(range(9)); random.shuffle(flat);
        current_inversions_parity = get_inversions(flat) % 2
        if current_inversions_parity == target_inversions_parity:
             state = [flat[i:i+3] for i in range(0, 9, 3)]
             # Final check ensures it's valid (should always be if generated correctly)
             if is_valid_state(state):
                 # print(f"Generated solvable state: {state}") # Debug
                 return state
        attempts += 1
    print(f"Error: Could not generate a solvable state after {max_attempts} attempts.")
    # Return the reference goal state as a list of lists if failed
    return [list(row) for row in goal_state_ref]

def get_solution_moves(path):
    """Determines the sequence of tile moves from a path of states."""
    if not path or len(path) < 2: return ["No moves in path."]
    moves = [];

    def find_blank(state):
         """Finds the (row, col) of the blank tile (0)."""
         for r, row in enumerate(state):
              for c, val in enumerate(row):
                   if val == 0: return r, c
         raise ValueError("Invalid state: Blank not found.")

    for i in range(len(path) - 1):
        s1, s2 = path[i], path[i+1]
        try:
            # Ensure states are valid before processing
            if not is_valid_state(s1) or not is_valid_state(s2):
                moves.append("Invalid state in path.")
                continue

            r1, c1 = find_blank(s1) # Blank pos in previous state
            r2, c2 = find_blank(s2) # Blank pos in current state

            # The tile that moved is the one at the blank's *previous* position (r1,c1)
            # in the *current* state s2.
            moved_tile_value = s2[r1][c1]

            move = f"Tile {moved_tile_value} moves "
            if r2 < r1: move += "Down"  # Blank moved up, so tile moved down
            elif r2 > r1: move += "Up"    # Blank moved down, so tile moved up
            elif c2 < c1: move += "Right" # Blank moved left, so tile moved right
            elif c2 > c1: move += "Left"  # Blank moved right, so tile moved left
            else: move += "Error? (No move)" # Should not happen for consecutive path states
            moves.append(move)
        except (ValueError, IndexError, TypeError) as e:
            moves.append(f"Error determining move ({e})")
            print(f"Error getting moves between:\nS1: {s1}\nS2: {s2}\nError: {e}") # Debug
            traceback.print_exc() # Print stack trace for detail
            continue
    return moves


# --- Lớp Puzzle (Core logic class - Unchanged structure) ---
class Puzzle:
    def __init__(self, start, goal=GOAL_STATE):
        # Ensure start is mutable list of lists if passed as tuple/immutable
        start_list = [list(row) for row in start]
        if not is_valid_state(start_list):
            raise ValueError("Invalid start state provided to Puzzle.")
        self.start = start_list
        self.goal = [list(row) for row in goal] # Ensure goal is mutable too
        self.goal_tuple = state_to_tuple(self.goal)
        self.rows = 3; self.cols = 3 # Fixed 3x3
        self._goal_pos_cache = self._build_goal_pos_cache()

        # --- Algorithm Parameter Attributes (Initialized with defaults) ---
        self.iddfs_max_depth = DEFAULT_MAX_DEPTH
        self.sa_initial_temp = 1000
        self.sa_cooling_rate = 0.997
        self.sa_min_temp = 0.1
        self.beam_width = 5
        self.ga_pop_size = 60
        self.ga_mutation_rate = 0.15
        self.ga_elite_size = 5
        self.ga_tournament_k = 3
        # Note: The general max_depth_limit isn't used by default in DFS/Backtrack
        # in this version, they use their own limits (MAX_DFS_POPS).
        # IDDFS and DLS Visit use self.iddfs_max_depth.

    def _build_goal_pos_cache(self):
        cache = {};
        for r in range(self.rows):
            for c in range(self.cols):
                if self.goal[r][c] != 0: cache[self.goal[r][c]] = (r, c)
        return cache

    def get_blank_position(self, state):
        # Accepts list of lists or tuple of tuples
        for r in range(self.rows):
            for c in range(self.cols):
                if state[r][c] == 0: return r, c
        raise ValueError("Invalid state: Blank (0) not found.")

    def is_goal(self, state):
        """Checks if the given state matches the puzzle's goal state."""
        return state_to_tuple(state) == self.goal_tuple

    def get_neighbors(self, state):
        """Generates valid neighbor states by moving the blank tile."""
        neighbors = [];
        try: r, c = self.get_blank_position(state)
        except ValueError: return neighbors # Return empty list if blank not found
        moves = [(0, 1), (0, -1), (1, 0), (-1, 0)] # R, L, D, U relative to blank
        current_state_list = [list(row) for row in state] # Ensure mutable copy

        for dr, dc in moves:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                # Create a distinct copy for the neighbor
                new_state = [row[:] for row in current_state_list]
                # Swap blank and adjacent tile
                new_state[r][c], new_state[nr][nc] = new_state[nr][nc], new_state[r][c]
                neighbors.append(new_state) # Append the new state (list of lists)
        return neighbors

    def heuristic(self, state): # Manhattan distance
        """Calculates the Manhattan distance heuristic for a state."""
        distance = 0
        # Convert to tuple for efficient checking if needed
        state_tuple = state if isinstance(state, tuple) else state_to_tuple(state)
        for r in range(self.rows):
            for c in range(self.cols):
                val = state_tuple[r][c]
                if val != 0:
                    if val in self._goal_pos_cache:
                        goal_r, goal_c = self._goal_pos_cache[val]
                        distance += abs(r - goal_r) + abs(c - goal_c)
                    else:
                         # This state can never reach the goal if a number is missing/wrong
                         return float('inf')
        return distance

    # --- Standard Search Algorithms (Unchanged Logic) ---
    def bfs(self):
        print("BFS: Starting..."); q=deque([(self.start,[self.start])]); v={state_to_tuple(self.start)}; c=0
        while q and c<MAX_BFS_STATES:
            s,p=q.popleft(); c+=1;
            if self.is_goal(s): print(f"BFS: Solved ({c} states)"); return p
            for n in self.get_neighbors(s):
                nt=state_to_tuple(n);
                if nt not in v: v.add(nt); q.append((n,p+[n]))
        st='Limit reached' if c>=MAX_BFS_STATES else 'Queue empty'; print(f"BFS: Failed/Limit ({st})"); return None

    def ucs(self):
        print("UCS: Starting...")
        start_tuple = state_to_tuple(self.start)
        priority_queue = [(0, start_tuple, [self.start])] # (cost, state_tuple, path_list_of_states)
        visited = {start_tuple: 0} # {state_tuple: min_cost_found}
        count = 0
        max_expansions = MAX_BFS_STATES # Reuse limit

        while priority_queue and count < max_expansions:
            cost, current_tuple, path = heapq.heappop(priority_queue)
            if cost > visited[current_tuple]: continue
            count += 1
            current_state = path[-1]
            if self.is_goal(current_state):
                print(f"UCS: Solved! Cost={cost}, Expanded={count}"); return path
            for neighbor_state in self.get_neighbors(current_state):
                neighbor_tuple = state_to_tuple(neighbor_state)
                new_cost = cost + 1
                if neighbor_tuple not in visited or new_cost < visited[neighbor_tuple]:
                    visited[neighbor_tuple] = new_cost
                    heapq.heappush(priority_queue, (new_cost, neighbor_tuple, path + [neighbor_state]))
        if count >= max_expansions: print(f"UCS: Failed/Limit ({max_expansions})")
        else: print("UCS: Failed (Queue empty)")
        return None

    def dfs(self):
        print("DFS: Starting..."); st=[(self.start,[self.start])]; v={state_to_tuple(self.start)}; c=0
        dfs_depth_limit = self.iddfs_max_depth + 15 # Allow deeper than strict IDDFS limit

        while st and c<MAX_DFS_POPS:
            s,p=st.pop(); c+=1;
            if self.is_goal(s): print(f"DFS: Solved ({c} pops)"); return p
            if len(p) > dfs_depth_limit: continue # Pruning based on depth
            for n in reversed(self.get_neighbors(s)):
                nt=state_to_tuple(n);
                if nt not in v: v.add(nt); st.append((n,p+[n]))
        st='Limit reached' if c>=MAX_DFS_POPS else 'Stack empty'; print(f"DFS: Failed/Limit ({st})"); return None

    def iddfs(self):
        print(f"IDDFS: Starting (Max Depth={self.iddfs_max_depth})...");
        s_t=state_to_tuple(self.start); nodes_total=0

        def dls(state, path, depth_limit, visited_in_path):
            nonlocal nodes_total; nodes_total += 1;
            if nodes_total % 100000 == 0: print(f"IDDFS: Visited ~{nodes_total}...")
            if self.is_goal(state): return path
            if depth_limit == 0: return None
            for neighbor in self.get_neighbors(state):
                neighbor_tuple = state_to_tuple(neighbor);
                if neighbor_tuple not in visited_in_path:
                    result = dls(neighbor, path + [neighbor], depth_limit - 1, visited_in_path | {neighbor_tuple});
                    if result: return result
            return None

        for depth in range(self.iddfs_max_depth + 1):
            print(f"IDDFS: Trying depth {depth}...")
            visited_this_depth = {s_t}
            result_path = dls(self.start, [self.start], depth, visited_this_depth)
            if result_path:
                print(f"IDDFS: Solved! (Depth={depth}, Nodes ~{nodes_total})"); return result_path
        print(f"IDDFS: Failed (Max Depth {self.iddfs_max_depth} reached, Nodes ~{nodes_total})"); return None


    def greedy(self):
        print("Greedy: Starting..."); s_t=state_to_tuple(self.start);
        pq=[(self.heuristic(self.start), s_t, [self.start])]; # (h, state_tuple, path)
        visited={s_t}; count=0
        while pq and count<MAX_GREEDY_EXPANSIONS:
            _, current_tuple, path = heapq.heappop(pq); count+=1
            current_state = path[-1]
            if self.is_goal(current_state): print(f"Greedy: Solved ({count} expansions)"); return path
            for neighbor_state in self.get_neighbors(current_state):
                neighbor_tuple=state_to_tuple(neighbor_state);
                if neighbor_tuple not in visited:
                    visited.add(neighbor_tuple);
                    h = self.heuristic(neighbor_state)
                    if h != float('inf'): heapq.heappush(pq,(h, neighbor_tuple, path+[neighbor_state]))
        st='Limit reached' if count>=MAX_GREEDY_EXPANSIONS else 'Queue empty'; print(f"Greedy: Failed/Limit ({st})"); return None

    def a_star(self):
        print("A*: Starting..."); s_t=state_to_tuple(self.start);
        # open_list: (f_cost, g_cost, state_list, state_tuple)
        open_list = [(self.heuristic(self.start) + 0, 0, self.start, s_t)]
        g_costs = {s_t: 0} # {state_tuple: cost_from_start}
        # came_from: {state_tuple: (predecessor_tuple, predecessor_state)}
        came_from = {s_t: (None, None)}
        closed_set = set(); count = 0
        max_expansions = MAX_IDA_STAR_NODES # Reuse limit

        while open_list and count < max_expansions:
            f, g, current_state, current_tuple = heapq.heappop(open_list)
            count += 1

            # Optimization: If we found a shorter path to this node already, skip processing it through this longer path
            if g > g_costs.get(current_tuple, float('inf')):
                 continue

            if current_tuple in closed_set: continue
            closed_set.add(current_tuple)

            if current_tuple == self.goal_tuple:
                # Reconstruct path using came_from which stores states
                path = []
                curr_t = current_tuple
                # Start from the goal state node we just found
                final_state_in_pq = current_state # The state associated with goal tuple in PQ
                path.append(final_state_in_pq)
                # Trace back predecessors
                pred_t, pred_s = came_from.get(curr_t, (None, None))
                while pred_t is not None:
                    if pred_s is None: # Should not happen if came_from populated correctly
                         print("A* Path Error: Predecessor state is None!")
                         break
                    path.append(pred_s)
                    curr_t = pred_t
                    pred_t, pred_s = came_from.get(curr_t, (None, None))
                # Add the start state if it wasn't added (it should be the last pred_s)
                if not path or state_to_tuple(path[-1]) != s_t:
                     if state_to_tuple(self.start) == s_t:
                          # print("A* Path: Adding start state explicitly.") # Debug
                          path.append(self.start)
                     else: print("A* Path Warning: Start state mismatch!")

                path.reverse()

                # Verification checks
                if not path or not is_valid_state(path[0]) or state_to_tuple(path[0]) != s_t:
                    print(f"A* Path Reconstruction Warning: Path doesn't start correctly. Got: {path[0] if path else 'Empty'}")
                if not path or not is_valid_state(path[-1]) or not self.is_goal(path[-1]):
                    print(f"A* Path Reconstruction Warning: Path doesn't end at goal. Got: {path[-1] if path else 'Empty'}")

                print(f"A*: Solved! (Expanded={count}, Cost={g})"); return path

            for neighbor_state in self.get_neighbors(current_state):
                neighbor_tuple = state_to_tuple(neighbor_state);
                if neighbor_tuple in closed_set: continue

                tentative_g = g + 1
                if tentative_g < g_costs.get(neighbor_tuple, float('inf')):
                    g_costs[neighbor_tuple] = tentative_g
                    came_from[neighbor_tuple] = (current_tuple, current_state) # Store pred tuple and state
                    h = self.heuristic(neighbor_state);
                    if h == float('inf'): continue # Skip states that can't reach goal

                    f_new = tentative_g + h;
                    heapq.heappush(open_list,(f_new, tentative_g, neighbor_state, neighbor_tuple))

        status = 'Limit reached' if count >= max_expansions else 'Queue empty'
        print(f"A*: Failed ({status})"); return None

    def ida_star(self):
        print("IDA*: Starting..."); start_tuple=state_to_tuple(self.start);
        bound = self.heuristic(self.start)
        path = [self.start]; # Store full states in path
        nodes_expanded = 0

        def search(current_path, g_cost, current_bound):
            nonlocal nodes_expanded; nodes_expanded += 1
            if nodes_expanded > MAX_IDA_STAR_NODES: raise MemoryError(f"IDA* Node limit ({MAX_IDA_STAR_NODES}) reached")

            current_state = current_path[-1];
            h_cost = self.heuristic(current_state)
            if h_cost == float('inf'): return False, float('inf')

            f_cost = g_cost + h_cost
            if f_cost > current_bound: return False, f_cost

            if self.is_goal(current_state): return True, current_path

            min_f_cost_above_bound = float('inf')
            current_path_tuples = frozenset(state_to_tuple(s) for s in current_path)

            for neighbor_state in self.get_neighbors(current_state):
                neighbor_tuple = state_to_tuple(neighbor_state)
                if neighbor_tuple not in current_path_tuples:
                    found, result = search(current_path + [neighbor_state], g_cost + 1, current_bound)
                    if found: return True, result
                    min_f_cost_above_bound = min(min_f_cost_above_bound, result)
            return False, min_f_cost_above_bound

        iteration = 0
        while True:
            iteration += 1; nodes_before_search = nodes_expanded
            print(f"IDA*: Iteration {iteration}, Bound={bound}...")
            try:
                found, result = search(path, 0, bound)
                nodes_expanded_this_iter = nodes_expanded - nodes_before_search
                print(f"IDA*: Iteration {iteration} finished. Nodes this iter: {nodes_expanded_this_iter}, Total nodes: {nodes_expanded}")

                if found:
                     print(f"IDA*: Solved! (Bound={bound}, Total Nodes ~{nodes_expanded})"); return result
                elif result == float('inf'):
                     print("IDA*: Failed (No solution possible)"); return None
                else:
                     if result <= bound :
                         print(f"IDA* Warning: New bound {result} <= old bound {bound}. Incrementing.")
                         bound += 1 # Ensure progress
                     else: bound = result
            except MemoryError as me: print(f"IDA*: Failed ({me})"); return None
            if bound > 100: print(f"IDA*: Stopping (Bound {bound} > 100)"); return None

    def beam_search(self, beam_width=None):
        bw = beam_width if beam_width is not None else self.beam_width
        print(f"Beam Search: Starting (Width={bw})...");
        start_tuple=state_to_tuple(self.start);
        # Beam: (heuristic, state_list, path_list_of_states)
        beam = [(self.heuristic(self.start), self.start, [self.start])];
        # Visited stores tuples seen in any beam to avoid re-adding duplicates in candidates
        visited = {start_tuple}; step = 0; best_goal_path = None

        while beam and step < MAX_BEAM_STEPS:
            step += 1; candidates = [];
            # Use a temporary set to track tuples added to candidates in this step
            candidates_tuples_this_step = set()

            for h, current_state, path in beam:
                if self.is_goal(current_state):
                    if best_goal_path is None or len(path) < len(best_goal_path):
                        print(f"Beam Search: Found goal step {step}, len {len(path)-1}")
                        best_goal_path = path
                    # Continue processing other nodes in the beam

                for neighbor_state in self.get_neighbors(current_state):
                    neighbor_tuple = state_to_tuple(neighbor_state);
                    # Add if not visited globally OR if it's the goal state (allow finding goal again)
                    # And also ensure it wasn't added to candidates *in this step* already
                    if (neighbor_tuple not in visited or self.is_goal(neighbor_state)) and \
                       neighbor_tuple not in candidates_tuples_this_step:
                         neighbor_h = self.heuristic(neighbor_state)
                         if neighbor_h != float('inf'):
                            candidates.append((neighbor_h, neighbor_state, path + [neighbor_state]));
                            visited.add(neighbor_tuple) # Add to global visited set
                            candidates_tuples_this_step.add(neighbor_tuple)

            if not candidates:
                 if best_goal_path: print(f"Beam Search: Solved (Stuck step {step})"); return best_goal_path
                 else: print(f"Beam Search: Failed (Stuck step {step})"); return None

            candidates.sort(key=lambda x: x[0]); beam = candidates[:bw]

        if best_goal_path: print(f"Beam Search: Solved (Max steps {step}, best len {len(best_goal_path)-1})"); return best_goal_path
        else: print(f"Beam Search: Failed/Limit (Max steps {step})"); return None

    # --- Backtracking and Exploration Algorithms ---
    def backtracking(self): # Returns exploration sequence
        print("Backtrack: Starting...")
        start_tuple = state_to_tuple(self.start)
        stack = [(self.start, [self.start])] # (state, path_to_state)
        visited = {start_tuple}; exploration_sequence = []; pops_count = 0; max_pops = MAX_DFS_POPS

        while stack and pops_count < max_pops:
            current_state, path = stack.pop(); pops_count += 1;
            exploration_sequence.append(current_state)
            if self.is_goal(current_state):
                print(f"Backtrack: Solved! ({pops_count} pops, path len {len(path)-1})");
                return exploration_sequence
            # Optional depth limit (can make it incomplete)
            # if len(path) > self.iddfs_max_depth + 20: continue
            for neighbor in reversed(self.get_neighbors(current_state)):
                neighbor_tuple = state_to_tuple(neighbor)
                if neighbor_tuple not in visited:
                     visited.add(neighbor_tuple); stack.append((neighbor, path + [neighbor]))
        status = 'Limit reached' if pops_count >= max_pops else 'Stack empty'
        print(f"Backtrack: Failed/Limit ({status}, {pops_count} pops explored)");
        return exploration_sequence

    def csp_solve(self): # Trả về chuỗi khám phá, giống Backtracking
        print("CSP Solve (via Backtrack): Starting...")
        start_tuple = state_to_tuple(self.start)
        stack = [(self.start, [self.start])] # (state, path_to_state)
        visited = {start_tuple}
        exploration_sequence = [] # <--- Thêm dòng này
        pops_count = 0
        max_pops = MAX_DFS_POPS # Sử dụng giới hạn tương tự DFS/Backtrack

        while stack and pops_count < max_pops:
            current_state, path = stack.pop()
            pops_count += 1
            exploration_sequence.append(current_state) # <--- Thêm trạng thái vào chuỗi khám phá

            if self.is_goal(current_state):
                 print(f"CSP Solve: Solved! ({pops_count} pops, path len {len(path)-1})")
                 # Trả về đường đi (path) vì nó cũng chính là chuỗi khám phá dẫn đến đích
                 # trong trường hợp này của DFS/Backtrack.
                 return path
                 # Hoặc nếu muốn luôn trả về toàn bộ những gì đã khám phá *cho đến khi* tìm thấy đích:
                 # return exploration_sequence # <--- Thay đổi nếu muốn xem toàn bộ khám phá

            # Optional depth limit (giữ nguyên nếu bạn muốn giới hạn độ sâu)
            # if len(path) > self.iddfs_max_depth + 20: continue

            # Thêm các hàng xóm chưa được thăm vào stack
            for neighbor in reversed(self.get_neighbors(current_state)): # Duyệt ngược để giống DFS
                neighbor_tuple = state_to_tuple(neighbor)
                if neighbor_tuple not in visited:
                    visited.add(neighbor_tuple)
                    stack.append((neighbor, path + [neighbor])) # Thêm vào stack

        status = 'Limit reached' if pops_count >= max_pops else 'Stack empty'
        print(f"CSP Solve: Failed/Limit ({status}, {pops_count} pops explored)")
        # Nếu không tìm thấy đích, trả về toàn bộ chuỗi khám phá
        return exploration_sequence # <--- Trả về chuỗi khám phá khi thất bại

    def and_or_search(self): # Treat as DLS Visit, return path if found
        max_depth = self.iddfs_max_depth # Use configured depth limit
        print(f"DLS Visit: Starting (Depth Limit={max_depth})...");
        visited_global = set()

        def dls_recursive(state, current_path_tuples, depth):
            nonlocal visited_global; state_tuple = state_to_tuple(state); visited_global.add(state_tuple)
            if self.is_goal(state): return []
            if depth <= 0: return None
            shortest_sub_path = None
            for neighbor in self.get_neighbors(state):
                neighbor_tuple = state_to_tuple(neighbor)
                if neighbor_tuple not in current_path_tuples:
                    sub_path = dls_recursive(neighbor, current_path_tuples | {neighbor_tuple}, depth - 1)
                    if sub_path is not None:
                        full_path_from_neighbor = [neighbor] + sub_path
                        if shortest_sub_path is None or len(full_path_from_neighbor) < len(shortest_sub_path):
                            shortest_sub_path = full_path_from_neighbor
            return shortest_sub_path

        start_tuple = state_to_tuple(self.start)
        found_sub_path = dls_recursive(self.start, frozenset({start_tuple}), max_depth)
        if found_sub_path is not None:
            full_path = [self.start] + found_sub_path
            print(f"DLS Visit: Solved! (Path Len {len(full_path)-1}, Limit {max_depth}, Visited {len(visited_global)})");
            return full_path
        else:
            print(f"DLS Visit: Failed (Limit {max_depth}, Visited {len(visited_global)})");
            return None

    # --- Local Search Algorithms ---
    def _hill_climbing_base(self, find_next_state_func, name):
        print(f"{name}: Starting..."); current_state = [r[:] for r in self.start]
        path_taken = [[r[:] for r in current_state]]; visited_tuples = {state_to_tuple(current_state)}; steps = 0
        while not self.is_goal(current_state) and steps < MAX_HILL_CLIMBING_STEPS:
            steps += 1; current_heuristic = self.heuristic(current_state)
            if current_heuristic == float('inf'):
                print(f"{name}: Stuck H=inf step {steps}. Aborting."); return path_taken
            next_state = find_next_state_func(current_state, current_heuristic)
            if next_state is None: print(f"{name}: Stuck step {steps}, H={current_heuristic}"); return path_taken
            next_tuple = state_to_tuple(next_state)
            if next_tuple in visited_tuples: print(f"{name}: Cycle step {steps}, H={self.heuristic(next_state)}"); return path_taken
            current_state = next_state; visited_tuples.add(next_tuple); path_taken.append([r[:] for r in current_state])
        if self.is_goal(current_state): print(f"{name}: Solved in {steps} steps.")
        else: print(f"{name}: Max steps ({MAX_HILL_CLIMBING_STEPS}) reached. H={self.heuristic(current_state)}")
        return path_taken

    def simple_hill_climbing(self):
        def find_first_better(state, current_h):
            neighbors = self.get_neighbors(state); random.shuffle(neighbors)
            for n_state in neighbors:
                h = self.heuristic(n_state);
                if h != float('inf') and h < current_h: return [r[:] for r in n_state]
            return None
        return self._hill_climbing_base(find_first_better,"Simple HC")

    def steepest_ascent_hill_climbing(self):
        def find_best_neighbor(state, current_h):
            best_neighbor = None; best_h = current_h
            for n_state in self.get_neighbors(state):
                h = self.heuristic(n_state)
                if h != float('inf') and h < best_h: best_h = h; best_neighbor = [r[:] for r in n_state]
            return best_neighbor
        return self._hill_climbing_base(find_best_neighbor,"Steepest HC")

    def stochastic_hill_climbing(self):
        def find_random_better(state, current_h):
            better_neighbors = []
            for n_state in self.get_neighbors(state):
                h = self.heuristic(n_state)
                if h != float('inf') and h < current_h: better_neighbors.append([r[:] for r in n_state])
            return random.choice(better_neighbors) if better_neighbors else None
        return self._hill_climbing_base(find_random_better,"Stochastic HC")

    def simulated_annealing(self, initial_temp=None, cooling_rate=None, min_temp=None):
        temp = initial_temp if initial_temp is not None else self.sa_initial_temp
        cool_rate = cooling_rate if cooling_rate is not None else self.sa_cooling_rate
        final_temp = min_temp if min_temp is not None else self.sa_min_temp
        print(f"SA: Starting (T0={temp:.2f}, Rate={cool_rate:.4f}, Tmin={final_temp:.2f})...");
        current_state=[r[:] for r in self.start]; current_h = self.heuristic(current_state);
        if current_h == float('inf'): print("SA: Invalid start H=inf."); return [current_state]
        path_taken = [[r[:] for r in current_state]]; iterations = 0; last_accepted_tuple=state_to_tuple(current_state)
        while temp > final_temp and iterations < MAX_SA_ITERATIONS:
            iterations += 1
            # if self.is_goal(current_state): print(f"SA: Solved iter {iterations}, T={temp:.2f}"); # Optional early exit
            neighbors = self.get_neighbors(current_state);
            if not neighbors: print(f"SA: Stuck iter {iterations}"); break
            next_state = random.choice(neighbors); next_h = self.heuristic(next_state);
            if next_h == float('inf'): continue
            delta_e = next_h - current_h; accept = False
            if delta_e < 0: accept = True
            else:
                 if temp > 1e-9:
                      try:
                          accept_prob = math.exp(-delta_e / temp)
                          if random.random() < accept_prob: accept = True
                      except OverflowError: accept = False
                 else: accept = False
            if accept:
                current_state = next_state; current_h = next_h; current_tuple = state_to_tuple(current_state)
                if current_tuple != last_accepted_tuple:
                    path_taken.append([r[:] for r in current_state]); last_accepted_tuple = current_tuple
            temp *= cool_rate
        final_h = self.heuristic(current_state); goal_reached = " (Goal)" if self.is_goal(current_state) else ""
        reason = "Min Temp" if temp <= final_temp else f"Max Iter ({MAX_SA_ITERATIONS})"
        print(f"SA: Finished ({reason}). Iter={iterations}, Final T={temp:.2f}, Final H={final_h}{goal_reached}"); return path_taken

    # --- Genetic Algorithm ---
    def genetic_algorithm_solve(self, population_size=None, mutation_rate=None, elite_size=None, tournament_k=None):
        pop_size = population_size if population_size is not None else self.ga_pop_size
        mut_rate = mutation_rate if mutation_rate is not None else self.ga_mutation_rate
        elite_s = elite_size if elite_size is not None else self.ga_elite_size
        tourn_k = tournament_k if tournament_k is not None else self.ga_tournament_k
        print(f"GA: Starting (Pop={pop_size}, Mut={mut_rate:.2f}, Elite={elite_s}, TournK={tourn_k})...");

        def state_to_flat(state): return sum(state, [])
        def flat_to_state(flat_list): return [flat_list[i:i+3] for i in range(0, 9, 3)] if len(flat_list)==9 else None

        population = []; attempts = 0
        while len(population) < pop_size and attempts < pop_size * 5:
             state = generate_random_solvable_state(self.goal_tuple);
             if state and is_valid_state(state): population.append(state)
             attempts += 1
        if len(population) < pop_size // 2: print(f"GA: Error generating pop (got {len(population)}). Aborting."); return None
        print(f"GA: Initial population size: {len(population)}")
        best_solution_overall = None; best_heuristic_overall = float('inf')

        for generation in range(MAX_GA_GENERATIONS):
            pop_fit = []
            for state in population:
                h = self.heuristic(state)
                if h == float('inf'): continue
                pop_fit.append({'state': state, 'heuristic': h})
                if h < best_heuristic_overall:
                    best_heuristic_overall = h; best_solution_overall = [r[:] for r in state]
                    if best_heuristic_overall == 0:
                        print(f"GA: Solved! Gen {generation}!"); return [best_solution_overall]
            if not pop_fit: print(f"GA: Error - invalid pop gen {generation}."); break
            pop_fit.sort(key=lambda x: x['heuristic'])

            next_population = [item['state'] for item in pop_fit[:min(elite_s, len(pop_fit))]]

            def tournament_selection(pf, k):
                if not pf: return None; k = min(k, len(pf));
                contenders = random.sample(pf, k)
                contenders.sort(key=lambda x: x['heuristic']); return contenders[0]['state']

            def cycle_crossover(p1_state, p2_state):
                p1_flat=state_to_flat(p1_state); p2_flat=state_to_flat(p2_state); size=len(p1_flat)
                child1_flat=[-1]*size; child2_flat=[-1]*size; cycles = []; visited_indices = [False] * size
                for i in range(size):
                    if not visited_indices[i]:
                        current_cycle = []; start_index = i; current_index = i
                        while not visited_indices[current_index]:
                            visited_indices[current_index] = True; current_cycle.append(current_index)
                            value_p2 = p2_flat[current_index]
                            try: current_index = p1_flat.index(value_p2)
                            except ValueError: print("GA CX Error: Value mismatch"); return p1_state, p2_state
                        cycles.append(current_cycle)
                for i, cycle in enumerate(cycles):
                    source1, source2 = (p1_flat, p2_flat) if i % 2 == 0 else (p2_flat, p1_flat)
                    for index in cycle:
                        if 0 <= index < size: child1_flat[index] = source1[index]; child2_flat[index] = source2[index]
                child1 = flat_to_state(child1_flat); child2 = flat_to_state(child2_flat);
                if not child1 or not is_valid_state(child1): child1 = p1_state
                if not child2 or not is_valid_state(child2): child2 = p2_state
                return child1, child2

            def mutate(state):
                mutated_state = [r[:] for r in state]; flat_list = state_to_flat(mutated_state)
                idx1, idx2 = random.sample(range(len(flat_list)), 2)
                flat_list[idx1], flat_list[idx2] = flat_list[idx2], flat_list[idx1];
                new_state = flat_to_state(flat_list)
                return new_state if new_state and is_valid_state(new_state) else state

            while len(next_population) < pop_size:
                parent1 = tournament_selection(pop_fit, tourn_k); parent2 = tournament_selection(pop_fit, tourn_k)
                if not parent1 or not parent2: continue
                child1, child2 = cycle_crossover(parent1, parent2)
                if random.random() < mut_rate: child1 = mutate(child1)
                if random.random() < mut_rate: child2 = mutate(child2)
                if child1 and is_valid_state(child1) and len(next_population) < pop_size: next_population.append(child1)
                if child2 and is_valid_state(child2) and len(next_population) < pop_size: next_population.append(child2)

            population = next_population
            if generation % 10 == 0 or generation == MAX_GA_GENERATIONS - 1:
                 print(f"GA Gen {generation}: Best H={best_heuristic_overall}, Pop Size={len(population)}")

        print(f"GA: Max generations ({MAX_GA_GENERATIONS}) reached. Best H: {best_heuristic_overall}")
        return [best_solution_overall] if best_solution_overall else None
# --- END Lớp Puzzle ---


# --- Pygame GUI Application ---
class PygamePuzzleApp:
    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("8-Puzzle AI Solver (Pygame)")
        self.clock = pygame.time.Clock()
        self.running = True

        # --- UI Element Storage ---
        self.buttons = {} # {name: {'rect': pygame.Rect, 'text': str, 'func': callable, 'enabled': bool, 'type': str}}
        self.sliders = {} # {name: {'rect': pygame.Rect, 'knob_rect': pygame.Rect, 'min': float, 'max': float, 'value': float, 'label': str, 'dragging': bool}}
        self.grid_rect = pygame.Rect(GRID_X, GRID_Y, GRID_SIZE, GRID_SIZE)
        self.results_rect = pygame.Rect(RESULTS_X, RESULTS_Y, RESULTS_WIDTH, RESULTS_HEIGHT)
        self.message_rect = pygame.Rect(CONTROLS_X, CONTROLS_Y, CONTROLS_WIDTH, 50) # Area for messages
        self.controls_rect = pygame.Rect(CONTROLS_X, CONTROLS_Y + 60, CONTROLS_WIDTH, HEIGHT - CONTROLS_Y - 70 - BOTTOM_MARGIN) # Area for buttons below message
        self.bottom_panel_rect = pygame.Rect(BOTTOM_PANEL_X, BOTTOM_PANEL_Y, BOTTOM_PANEL_WIDTH, BOTTOM_PANEL_HEIGHT)

        # --- Application State Variables ---
        self.app_state = "INPUT" # INPUT, READY, FAILED, PLACING_RANDOM, RUNNING, VISUALIZING, BENCHMARKING, SOLVED
        self.current_grid_state = [[0] * 3 for _ in range(3)]
        self.num_to_place = 1
        self.user_start_state = None
        self.selected_algorithm_name = None
        self.selected_algorithm_text = None
        self.solution_path = None
        self.vis_step_index = 0
        self.puzzle_instance = None # Will hold the Puzzle object when needed
        self.message = ""
        self.message_color = MSG_DEFAULT_COLOR
        self.final_state_after_vis = None # Store expected state after visualization
        self.placing_target_state = None
        self.placing_current_number = 0
        self.result_queue = queue.Queue() # For thread communication
        self.results_lines = [] # Store lines for the results text area
        self.results_scroll_offset = 0 # How many pixels scrolled down
        self.results_total_height = 0 # Calculated total height of results text
        self.results_line_height = RESULTS_FONT.get_linesize()

        # --- Visualization Timing ---
        self.vis_delay = 175 # Milliseconds between visualization steps
        self.last_vis_time = 0
        self.placement_delay = 200 # ms for random placement animation
        self.last_placement_time = 0

        # --- Mouse/Interaction State ---
        self.hovered_button_name = None
        self.active_slider_name = None # Which slider knob is being dragged

        # --- Button Configuration (Similar to Tkinter version) ---
        self.buttons_config = {
            'BFS': {'func_name': 'bfs', 'type': 'path'}, 'UCS': {'func_name': 'ucs', 'type': 'path'},
            'DFS': {'func_name': 'dfs', 'type': 'path'}, 'IDDFS': {'func_name': 'iddfs', 'type': 'path'},
            'Greedy': {'func_name': 'greedy', 'type': 'path'}, 'A*': {'func_name': 'a_star', 'type': 'path'},
            'IDA*': {'func_name': 'ida_star', 'type': 'path'},
            'Backtrack': {'func_name': 'backtracking', 'type': 'generate_explore'},
            # --- THAY ĐỔI DÒNG NÀY ---
            'CSP Solve': {'func_name': 'csp_solve', 'type': 'generate_explore'}, # Đổi type thành 'generate_explore'
            # --- KẾT THÚC THAY ĐỔI ---
            'and_or_search': {'func_name': 'and_or_search', 'type': 'path_if_found'},
            'Simple HC': {'func_name': 'simple_hill_climbing', 'type': 'local'},
            'Steepest HC': {'func_name': 'steepest_ascent_hill_climbing', 'type': 'local'},
            'Stoch HC': {'func_name': 'stochastic_hill_climbing', 'type': 'local'},
            'Sim Anneal': {'func_name': 'simulated_annealing', 'type': 'local'},
            'Beam Srch': {'func_name': 'beam_search', 'type': 'path'},
            'Genetic': {'func_name': 'genetic_algorithm_solve', 'type': 'state_only'},
            'Benchmark': {'func_name': 'benchmark', 'type': 'action'},
            'Reset': {'func_name': 'reset', 'type': 'action'},
        }
        self.pathfinding_algos_for_benchmark = [
            'bfs', 'ucs', 'dfs', 'iddfs', 'greedy', 'a_star', 'ida_star', 'beam_search' # Add others if desired
        ]
        self.algo_name_map = {data['func_name']: text for text, data in self.buttons_config.items()}

        self._create_ui_elements()
        self.reset_app() # Initialize state and UI

    def _create_ui_elements(self):
        """Define the Rects and initial state for buttons and sliders."""
        # --- Buttons ---
        button_width = (CONTROLS_WIDTH - PANEL_PADDING * 1.5) // 2 # 2 columns
        button_height = 35
        col1_x = self.controls_rect.x + PANEL_PADDING // 2
        col2_x = col1_x + button_width + PANEL_PADDING // 2
        current_y = self.controls_rect.y + 5

        algo_buttons = []; action_buttons = []
        for text, config in self.buttons_config.items():
            if config['type'] == 'action': action_buttons.append((text, config))
            else: algo_buttons.append((text, config))

        col = 0
        for text, config in algo_buttons:
            x = col1_x if col == 0 else col2_x
            rect = pygame.Rect(x, current_y, button_width, button_height)
            self.buttons[text] = {'rect': rect, 'text': text, 'func_name': config['func_name'], 'type': config['type'], 'enabled': False}
            col += 1
            if col >= 2: col = 0; current_y += button_height + 5

        reset_y = self.controls_rect.bottom - button_height - 5
        bench_y = reset_y - button_height - 5
        reset_cfg = self.buttons_config['Reset']; reset_rect = pygame.Rect(col1_x, reset_y, self.controls_rect.width - PANEL_PADDING , button_height)
        self.buttons['Reset'] = {'rect': reset_rect, 'text': 'Reset', 'func_name': reset_cfg['func_name'], 'type': reset_cfg['type'], 'enabled': True}
        bench_cfg = self.buttons_config['Benchmark']; bench_rect = pygame.Rect(col1_x, bench_y, self.controls_rect.width - PANEL_PADDING, button_height)
        self.buttons['Benchmark'] = {'rect': bench_rect, 'text': 'Benchmark', 'func_name': bench_cfg['func_name'], 'type': bench_cfg['type'], 'enabled': False}

        # --- Sliders ---
        slider_section_width = BOTTOM_PANEL_WIDTH // 2 - PANEL_PADDING * 1.5 # Divide bottom panel
        slider_width = slider_section_width - 10 # Width of the track itself
        slider_height = 15; knob_width = 10; knob_height = 20
        slider_start_x = self.bottom_panel_rect.x + PANEL_PADDING
        slider_y = self.bottom_panel_rect.y + 10

        # Vis Speed Slider
        vis_rect = pygame.Rect(slider_start_x, slider_y, slider_width, slider_height)
        vis_knob_rect = pygame.Rect(0, 0, knob_width, knob_height)
        self.sliders['vis_speed'] = {'rect': vis_rect, 'knob_rect': vis_knob_rect, 'min': 10, 'max': 1000, 'value': 150, 'label': "Vis Speed(ms)", 'dragging': False}
        self.vis_delay = self.sliders['vis_speed']['value']

        # Max Depth Slider
        depth_slider_y = slider_y + knob_height + 10 # Place below first slider
        depth_rect = pygame.Rect(slider_start_x, depth_slider_y, slider_width, slider_height)
        depth_knob_rect = pygame.Rect(0, 0, knob_width, knob_height)
        self.sliders['max_depth'] = {'rect': depth_rect, 'knob_rect': depth_knob_rect, 'min': 1, 'max': 30, 'value': DEFAULT_MAX_DEPTH, 'label': "Max Depth", 'dragging': False}

        # Update knob positions based on initial values
        self._update_knob_position('vis_speed')
        self._update_knob_position('max_depth')

    # --- Drawing Functions ---
    def draw_grid(self):
        pygame.draw.rect(self.screen, DARK_GREY, self.grid_rect, border_radius=5)
        for r in range(3):
            for c in range(3):
                num = self.current_grid_state[r][c]
                tile_rect = pygame.Rect(GRID_X + c*TILE_SIZE + 2, GRID_Y + r*TILE_SIZE + 2, TILE_SIZE-4, TILE_SIZE-4)
                if num == 0: pygame.draw.rect(self.screen, TILE_EMPTY_BG, tile_rect, border_radius=3)
                elif num == -1: pygame.draw.rect(self.screen, DARK_RED, tile_rect, border_radius=3) # Error indication
                else:
                    color_index = (num - 1) % len(TILE_COLORS)
                    pygame.draw.rect(self.screen, TILE_COLORS[color_index], tile_rect, border_radius=3)
                    text_surf = TILE_FONT.render(str(num), True, WHITE)
                    text_rect = text_surf.get_rect(center=tile_rect.center)
                    self.screen.blit(text_surf, text_rect)
                pygame.draw.rect(self.screen, TILE_BORDER, tile_rect, 1, border_radius=3)

    def draw_buttons(self):
        mouse_pos = pygame.mouse.get_pos()
        self.hovered_button_name = None
        for name, data in self.buttons.items():
            rect = data['rect']; enabled = data['enabled']
            is_hovered = rect.collidepoint(mouse_pos) and enabled
            if not enabled: bg_color, text_color = BUTTON_DISABLED_COLOR, BUTTON_DISABLED_TEXT_COLOR
            elif is_hovered: bg_color, text_color = BUTTON_HOVER_COLOR, BUTTON_TEXT_COLOR; self.hovered_button_name = name
            else: bg_color, text_color = BUTTON_COLOR, BUTTON_TEXT_COLOR
            pygame.draw.rect(self.screen, bg_color, rect, border_radius=5)
            text_surf = BUTTON_FONT.render(data['text'], True, text_color)
            text_rect = text_surf.get_rect(center=rect.center)
            self.screen.blit(text_surf, text_rect)
            pygame.draw.rect(self.screen, DARK_GREY, rect, 1, border_radius=5)

    def draw_message(self):
        if self.message:
            lines = textwrap.wrap(self.message, width=35)
            y_offset = self.message_rect.y + 5
            for i, line in enumerate(lines):
                 if y_offset + MSG_FONT.get_linesize() > self.message_rect.bottom : break
                 msg_surf = MSG_FONT.render(line, True, self.message_color)
                 msg_rect = msg_surf.get_rect(centerx=self.message_rect.centerx, top=y_offset)
                 if msg_rect.width > self.message_rect.width - 10: msg_rect.left = self.message_rect.left + 5
                 self.screen.blit(msg_surf, msg_rect)
                 y_offset += MSG_FONT.get_linesize()

    def draw_results(self):
        pygame.draw.rect(self.screen, WHITE, self.results_rect)
        pygame.draw.rect(self.screen, DARK_GREY, self.results_rect, 1)
        y = self.results_rect.y + 5 - self.results_scroll_offset
        max_y = self.results_rect.bottom - 5
        original_clip = self.screen.get_clip()
        self.screen.set_clip(self.results_rect.inflate(-4, -4)) # Clip strictly inside border
        self.results_total_height = 0
        for i, line in enumerate(self.results_lines):
             line_height = self.results_line_height
             self.results_total_height += line_height
             if y + line_height > self.results_rect.top and y < max_y:
                 try:
                      res_surf = RESULTS_FONT.render(line, True, TEXT_COLOR)
                      res_rect = res_surf.get_rect(topleft=(self.results_rect.x + 5, y))
                      self.screen.blit(res_surf, res_rect)
                 except Exception as e: print(f"Error rendering line '{line}': {e}")
             y += line_height
             if y >= max_y + self.results_scroll_offset : break
        self.screen.set_clip(original_clip)
        # Draw Scrollbar
        visible_height = self.results_rect.height
        if self.results_total_height > visible_height:
            scrollbar_width = 10
            scrollbar_area_rect = pygame.Rect(self.results_rect.right - scrollbar_width - 2, self.results_rect.y + 2, scrollbar_width, self.results_rect.height - 4)
            pygame.draw.rect(self.screen, LIGHT_GREY, scrollbar_area_rect, border_radius=3)
            handle_height = max(15, visible_height * (visible_height / self.results_total_height))
            scroll_ratio = self.results_scroll_offset / (self.results_total_height - visible_height) if (self.results_total_height - visible_height) > 0 else 0
            handle_y = scrollbar_area_rect.y + scroll_ratio * (scrollbar_area_rect.height - handle_height)
            handle_rect = pygame.Rect(scrollbar_area_rect.x, handle_y, scrollbar_width, handle_height)
            pygame.draw.rect(self.screen, DARK_GREY, handle_rect, border_radius=3)

    def _update_knob_position(self, slider_name):
         if slider_name not in self.sliders: return
         data = self.sliders[slider_name]; track_rect = data['rect']; knob_rect = data['knob_rect']
         # Ensure max > min to avoid division by zero
         range_val = data['max'] - data['min']
         if range_val <= 0: val_ratio = 0.0
         else: val_ratio = (data['value'] - data['min']) / range_val
         knob_x = track_rect.x + val_ratio * (track_rect.width - knob_rect.width)
         knob_rect.centery = track_rect.centery; knob_rect.x = knob_x

    def _update_slider_value_from_pos(self, slider_name, mouse_x):
        if slider_name not in self.sliders: return False
        data = self.sliders[slider_name]; track_rect = data['rect']; knob_rect = data['knob_rect']
        relative_x = mouse_x - track_rect.x - (knob_rect.width / 2)
        track_eff_width = track_rect.width - knob_rect.width
        if track_eff_width <= 0: val_ratio = 0.0
        else: val_ratio = relative_x / track_eff_width
        val_ratio = max(0.0, min(1.0, val_ratio))
        new_value = data['min'] + val_ratio * (data['max'] - data['min'])
        old_value = data['value']
        # Round based on slider type
        if slider_name == 'max_depth': new_value = int(round(new_value))
        elif slider_name == 'vis_speed': new_value = int(round(new_value))
        # Add other sliders here (SA, Beam, GA params) with rounding
        if abs(new_value - old_value) > 1e-6: # Check if value actually changed
             data['value'] = new_value
             self._update_knob_position(slider_name)
             # Update corresponding parameter
             if slider_name == 'vis_speed': self.vis_delay = int(data['value'])
             elif slider_name == 'max_depth':
                  current_depth = int(data['value'])
                  if self.puzzle_instance: self.puzzle_instance.iddfs_max_depth = current_depth
                  # Update the global default for future puzzles too
                  global DEFAULT_MAX_DEPTH; DEFAULT_MAX_DEPTH = current_depth
                  print(f"Max Depth set to: {current_depth}") # Feedback
             # elif slider_name == 'beam_width': ...
             return True
        return False

    def draw_sliders(self):
        for name, data in self.sliders.items():
            track_rect = data['rect']; knob_rect = data['knob_rect']
            pygame.draw.rect(self.screen, SLIDER_TRACK_COLOR, track_rect, border_radius=5)
            pygame.draw.rect(self.screen, SLIDER_KNOB_COLOR, knob_rect, border_radius=3)
            pygame.draw.rect(self.screen, SLIDER_KNOB_BORDER, knob_rect, 1, border_radius=3)
            label_text = f"{data['label']}: {data['value']:.0f}"
            label_surf = SLIDER_FONT.render(label_text, True, TEXT_COLOR)
            # Position label to the right of the track
            label_rect = label_surf.get_rect(midleft=(track_rect.right + 10, track_rect.centery))
            # If label goes off-screen, try placing it above
            if label_rect.right > WIDTH - PANEL_PADDING:
                 label_rect.midbottom = (track_rect.centerx, track_rect.top - 2)
            self.screen.blit(label_surf, label_rect)

    def draw_notes(self):
        note_y = self.bottom_panel_rect.y + 5
        # Start notes in the second half of the bottom panel
        note_x = self.bottom_panel_rect.x + self.bottom_panel_rect.width // 2 + PANEL_PADDING
        notes = [
            "'Benchmark' compares pathfinders.",
            "'Reset' clears board & results.",
            "'Backtrack' generates & solves.",
            "Use sliders for Vis Speed & Depth.",
        ]
        for note in notes:
             if note_y + NOTE_FONT.get_linesize() > self.bottom_panel_rect.bottom: break
             note_surf = NOTE_FONT.render(note, True, NOTE_COLOR)
             note_rect = note_surf.get_rect(topleft=(note_x, note_y))
             # Prevent notes going off screen right
             if note_rect.right > WIDTH - PANEL_PADDING: continue
             self.screen.blit(note_surf, note_rect)
             note_y += NOTE_FONT.get_linesize() + 2

    # --- UI Update Methods ---
    def set_message(self, text, color=MSG_DEFAULT_COLOR):
        self.message = text; self.message_color = color
        print(f"MSG: {text}") # Debug

    def set_results(self, lines):
        self.results_lines = lines[:]; self.results_scroll_offset = 0
        self.results_total_height = sum(self.results_line_height for _ in lines)

    def update_grid_display(self, state):
         """Cập nhật self.current_grid_state để phản ánh trạng thái được cung cấp.
            Hàm này KHÔNG nên thực hiện việc xác thực is_valid_state.
            Nó chỉ hiển thị dữ liệu được cung cấp.
         """
         try:
             # Tạo một bản sao sâu (deep copy) để đảm bảo self.current_grid_state
             # không bị ảnh hưởng bởi các thay đổi bên ngoài đối với 'state'.
             # Quan trọng nếu 'state' đến từ solution_path hoặc các nguồn khác.
             new_state_copy = [list(row) for row in state]

             # Kiểm tra cấu trúc cơ bản (phải là 3x3)
             if len(new_state_copy) != 3 or any(len(r) != 3 for r in new_state_copy):
                 print(f"Warning: State provided to update_grid_display is not 3x3: {state}")
                 # Quyết định cách xử lý: hiển thị lỗi hoặc không làm gì cả.
                 # Hiển thị trạng thái lỗi (-1) có thể hữu ích để debug.
                 self.current_grid_state = [[-1]*3 for _ in range(3)]
                 # Bạn có thể muốn đặt một thông báo lỗi ở đây nếu điều này xảy ra bất ngờ
                 # self.set_message("Internal Error: Invalid display state format!", MSG_ERROR_COLOR)
             else:
                 self.current_grid_state = new_state_copy

         except (TypeError, IndexError) as e:
             print(f"Error updating grid display with state: {state}. Error: {e}")
             # Đặt trạng thái lỗi nếu có vấn đề khi sao chép hoặc cấu trúc không đúng
             self.current_grid_state = [[-1]*3 for _ in range(3)]
             self.set_message(f"Internal Error: Grid display update failed.", MSG_ERROR_COLOR)

         # Không cần gọi is_valid_state ở đây nữa.
         # if not is_valid_state(state):
         #    self.set_message("Error: Invalid grid state!", MSG_ERROR_COLOR) # Di chuyển logic này đi chỗ khác
         #    self.current_grid_state = [[-1]*3 for _ in range(3)] # Không ghi đè ở đây
         # else:
         #     self.current_grid_state = [list(row) for row in state] # Đã thực hiện ở trên

    def update_button_states(self):
        # Assume state is valid unless proven otherwise (simplifies initial check)
        start_state_is_ready = False
        if self.user_start_state:
            try: start_state_is_ready = is_solvable(self.user_start_state)
            except Exception as e: print(f"Error checking solvability for button update: {e}") # Catch potential errors during check

        can_interact = self.app_state not in ["PLACING_RANDOM", "RUNNING", "VISUALIZING", "BENCHMARKING"]

        for name, data in self.buttons.items():
            enabled = False; func_name = data['func_name']; algo_type = data['type']
            if can_interact:
                if algo_type in ['path', 'explore_path', 'path_if_found', 'local']: enabled = start_state_is_ready
                elif algo_type in ['generate_explore', 'state_only', 'action']:
                    enabled = True
                    if func_name == 'benchmark': enabled = start_state_is_ready
            if func_name == 'reset': enabled = can_interact
            data['enabled'] = enabled

    # --- Event Handling ---
    def handle_input(self):
        mouse_pos = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT: self.running = False
            if event.type == pygame.MOUSEWHEEL:
                if self.results_rect.collidepoint(mouse_pos):
                     scroll_amount = event.y * self.results_line_height * 2
                     max_scroll = max(0, self.results_total_height - self.results_rect.height)
                     self.results_scroll_offset -= scroll_amount
                     self.results_scroll_offset = max(0, min(self.results_scroll_offset, max_scroll))
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.grid_rect.collidepoint(mouse_pos) and self.app_state == "INPUT": self._handle_grid_click(mouse_pos)
                    elif self.hovered_button_name:
                        button_data = self.buttons[self.hovered_button_name]
                        if button_data['enabled']: self._handle_button_click(button_data['func_name'], self.hovered_button_name, button_data['type'])
                    else: # Check slider interaction last
                        for name, data in self.sliders.items():
                             if data['knob_rect'].collidepoint(mouse_pos) or data['rect'].collidepoint(mouse_pos):
                                 self.active_slider_name = name; data['dragging'] = True
                                 self._update_slider_value_from_pos(name, mouse_pos[0]) # Update on click too
                                 break
            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    if self.active_slider_name:
                        self.sliders[self.active_slider_name]['dragging'] = False; self.active_slider_name = None
            if event.type == pygame.MOUSEMOTION:
                 if self.active_slider_name and self.sliders[self.active_slider_name]['dragging']: self._update_slider_value_from_pos(self.active_slider_name, mouse_pos[0])
                 else: # Update button hover state if not dragging slider
                      self.hovered_button_name = None
                      for name, data in self.buttons.items():
                           if data['rect'].collidepoint(mouse_pos) and data['enabled']: self.hovered_button_name = name; break

    def _handle_grid_click(self, mouse_pos):
        if self.app_state != "INPUT": return
        col = (mouse_pos[0] - GRID_X) // TILE_SIZE
        row = (mouse_pos[1] - GRID_Y) // TILE_SIZE
        # Clamp row/col just in case calculation is slightly off at edges
        row = max(0, min(2, row))
        col = max(0, min(2, col))

        # Check if the cell is currently empty (contains 0)
        if self.current_grid_state[row][col] == 0:
            # Check if we still need to place numbers 1-8
            if self.num_to_place <= 8:
                self.current_grid_state[row][col] = self.num_to_place
                # print(f"Placed {self.num_to_place} at ({row}, {col})") # Debugging print
                placed_number = self.num_to_place # Lưu lại số vừa đặt
                self.num_to_place += 1
                self.update_grid_display(self.current_grid_state) # Update display after each placement

                # Check if we just placed the 8th number (meaning num_to_place is now 9)
                if self.num_to_place == 9:
                    # The grid is now visually complete (1-8 placed, one 0 remaining)
                    # Finalize the user's start state
                    self.user_start_state = [r[:] for r in self.current_grid_state]
                    # print(f"Input complete. Final state: {self.user_start_state}") # Debug

                    # --- VALIDATION MOVED HERE ---
                    # NOW check validity and solvability on the complete state
                    if is_valid_state(self.user_start_state): # First, ensure it's valid (should be)
                        if is_solvable(self.user_start_state):
                            self.set_message("Board ready! Select algorithm.", MSG_SUCCESS_COLOR)
                            self.app_state = "READY"
                        else:
                            # Mark state as unsolvable visually maybe? Optional.
                            self.set_message("Board is unsolvable! Reset.", MSG_WARN_COLOR)
                            self.app_state = "FAILED"
                    else:
                        # This case indicates a deeper logic error if reached
                        self.set_message("Error: Final input state is invalid!", MSG_ERROR_COLOR)
                        # print(f"INVALID FINAL STATE: {self.user_start_state}") # Debug
                        self.app_state = "FAILED"
                    # --- END OF MOVED VALIDATION ---

                    self.update_button_states() # Update buttons based on READY/FAILED state
                else:
                    # Still placing numbers 1-8
                    self.set_message(f"Placed {placed_number}. Click empty cell for {self.num_to_place}.", MSG_DEFAULT_COLOR)
            else:
                # This case should technically not be reached if app_state becomes READY/FAILED correctly
                # It means num_to_place > 8, board should be full
                self.set_message("Board full. Select algorithm or Reset.", MSG_INFO_COLOR)
        else:
            # Clicked on an already occupied cell
            occupied_by = self.current_grid_state[row][col]
            if self.num_to_place <= 8:
                self.set_message(f"Cell occupied by {occupied_by}! Click an empty cell for {self.num_to_place}.", MSG_WARN_COLOR)
            else:
                # Board is full, clicking occupied cells is irrelevant
                self.set_message(f"Board full (cell has {occupied_by}). Select algorithm or Reset.", MSG_INFO_COLOR)

    def _handle_button_click(self, func_name, text, algo_type):
        print(f"Button Clicked: {text} (func: {func_name}, type: {algo_type})")
        if func_name == 'reset': self.reset_app(); return
        if func_name == 'benchmark':
            if self.user_start_state and is_solvable(self.user_start_state): self.run_benchmark_threaded()
            else: self.set_message("Board not ready/unsolvable for Benchmark!", MSG_WARN_COLOR)
            return
        if algo_type == 'generate_explore': self.start_random_placement(); return

        start_s = None; puzzle_params = {}
        if algo_type in ['path', 'local', 'path_if_found', 'explore_path']:
            if self.user_start_state and is_solvable(self.user_start_state): start_s = [r[:] for r in self.user_start_state]
            else: self.set_message("Error: Board not set or unsolvable!", MSG_ERROR_COLOR); return
        elif algo_type == 'state_only': # Genetic
            self.set_message(f"{text} generating state...", MSG_INFO_COLOR); start_s = generate_random_solvable_state()
            if not start_s: self.set_message("Error generating random state for GA!", MSG_ERROR_COLOR); return
            self.update_grid_display(start_s); self.draw(); pygame.display.flip(); time.sleep(0.1)
        else: self.set_message(f"Error: Unknown button type '{algo_type}'!", MSG_ERROR_COLOR); return

        # Gather parameters from sliders
        if func_name in ['iddfs', 'and_or_search']:
             puzzle_params['iddfs_max_depth'] = int(self.sliders['max_depth']['value'])
        # Add other params (SA, Beam, GA) if sliders exist

        if start_s: self.run_algorithm_threaded(func_name, text, start_s, puzzle_params)
        else: self.set_message("Error: No valid start state.", MSG_ERROR_COLOR)

    # --- Core Application Logic ---
    def reset_app(self):
        print("\n--- Resetting Application ---")
        while not self.result_queue.empty():
            try: self.result_queue.get_nowait()
            except queue.Empty: break
        self.current_grid_state = [[0] * 3 for _ in range(3)]
        self.num_to_place = 1; self.user_start_state = None
        self.selected_algorithm_name = None; self.selected_algorithm_text = None
        self.solution_path = None; self.vis_step_index = 0
        self.puzzle_instance = None; self.final_state_after_vis = None
        self.placing_target_state = None; self.placing_current_number = 0
        self.results_lines = []; self.results_scroll_offset = 0; self.results_total_height = 0
        self.app_state = "INPUT"
        self.set_message("Click empty grid cell to place number 1.", MSG_DEFAULT_COLOR)
        self.update_grid_display(self.current_grid_state)
        self.update_button_states()

    def start_random_placement(self):
        self.set_message("Generating random board...", MSG_INFO_COLOR); self.set_results(["Generating board..."])
        self.current_grid_state = [[0] * 3 for _ in range(3)]; self.update_grid_display(self.current_grid_state)
        self.app_state = "PLACING_RANDOM"; self.update_button_states(); self.draw(); pygame.display.flip()
        self.placing_target_state = generate_random_solvable_state()
        if not self.placing_target_state:
            self.set_message("Error generating random state!", MSG_ERROR_COLOR); self.app_state = "FAILED"; self.update_button_states(); return
        self.placing_current_number = 1; self.last_placement_time = pygame.time.get_ticks()

    def _update_placement_animation(self):
        if self.app_state != "PLACING_RANDOM" or not self.placing_target_state: return
        now = pygame.time.get_ticks()
        if now - self.last_placement_time > self.placement_delay:
            self.last_placement_time = now
            if 1 <= self.placing_current_number <= 8:
                found_pos = False; target_r, target_c = -1, -1
                for r in range(3):
                    for c in range(3):
                        if self.placing_target_state[r][c] == self.placing_current_number: target_r, target_c = r, c; found_pos = True; break
                    if found_pos: break
                if found_pos:
                    self.current_grid_state[target_r][target_c] = self.placing_current_number
                    self.update_grid_display(self.current_grid_state)
                    self.set_message(f"Placing number {self.placing_current_number}...", MSG_INFO_COLOR)
                    self.placing_current_number += 1
                else:
                    print(f"Error: Could not find pos for {self.placing_current_number}."); self.set_message("Error generating board!", MSG_ERROR_COLOR)
                    self.app_state = "FAILED"; self.update_button_states(); self.placing_target_state = None
            else: # Finished 1-8
                print("Placement complete. Starting Backtrack.");
                try: # Ensure blank is correct
                    br, bc = Puzzle(self.placing_target_state).get_blank_position(self.placing_target_state)
                    if self.current_grid_state[br][bc] != 0:
                         for r0, row0 in enumerate(self.current_grid_state):
                              try: c0 = row0.index(0); self.current_grid_state[r0][c0] = self.placing_target_state[r0][c0]; break
                              except ValueError: continue
                         self.current_grid_state[br][bc] = 0
                    self.update_grid_display(self.current_grid_state)
                except Exception as e: print(f"Warn: Could not place blank: {e}")
                self.run_algorithm_threaded('backtracking', 'Backtrack', self.placing_target_state)
                self.placing_target_state = None; self.placing_current_number = 0

    def run_algorithm_threaded(self, algo_func_name, algo_display_text, start_state_input, puzzle_params={}):
        print(f"\n--- Preparing to run {algo_display_text} ---")
        self.selected_algorithm_name = algo_func_name; self.selected_algorithm_text = algo_display_text
        self.app_state = "RUNNING"; self.update_button_states()
        self.set_message(f"Running {algo_display_text}...", MSG_INFO_COLOR); self.set_results([f"Running {algo_display_text}...", "Please wait..."])
        self.solution_path = None; self.vis_step_index = 0; self.final_state_after_vis = "FAILED"
        try:
            self.puzzle_instance = Puzzle(start_state_input, GOAL_STATE)
            for param, value in puzzle_params.items():
                 if hasattr(self.puzzle_instance, param): setattr(self.puzzle_instance, param, value); print(f"  Set Puzzle param: {param} = {value}")
                 else: print(f"  Warn: Puzzle no attr '{param}'")
        except Exception as e: self.set_message(f"Error creating puzzle: {e}", MSG_ERROR_COLOR); self.app_state = "FAILED"; self.update_button_states(); return
        thread = threading.Thread(target=self._solver_thread_func, args=(self.puzzle_instance, algo_func_name, algo_display_text, self.result_queue), daemon=True)
        thread.start()

    # Solver thread function remains the same as previous version
    def _solver_thread_func(self, puzzle_obj, algo_func_name, algo_display_text, q):
        result_data = {'status': 'error', 'message': 'Unknown thread error', 'path': None, 'time': 0, 'algo_name': algo_func_name, 'algo_text': algo_display_text}
        try:
            solver_method = getattr(puzzle_obj, algo_func_name, None)
            if solver_method:
                t_start = time.perf_counter()
                try:
                    temp_result = solver_method()
                    t_elapsed = time.perf_counter() - t_start
                    result_data['time'] = t_elapsed; result_data['path'] = temp_result
                    if temp_result is not None and isinstance(temp_result, list) and len(temp_result) > 0:
                        algo_type = self.buttons_config[algo_display_text]['type']
                        is_goal_reached = False
                        try: final_state = temp_result[-1]; is_goal_reached = puzzle_obj.is_goal(final_state)
                        except: pass # Ignore errors checking goal in thread if state invalid
                        if is_goal_reached: result_data['status'] = 'success_goal'; result_data['message'] = f"{algo_display_text} solved."
                        else: result_data['status'] = 'success_nogoal'; result_data['message'] = f"{algo_display_text} finished (no goal)."
                    elif temp_result is None or (isinstance(temp_result, list) and not temp_result):
                        result_data['status'] = 'failure'; result_data['message'] = f"{algo_display_text} failed/found nothing."
                    else: result_data['status'] = 'error_type'; result_data['message'] = f"{algo_display_text}: Bad result type."
                except MemoryError as me: t_elapsed=time.perf_counter()-t_start; result_data['time']=t_elapsed; result_data['status']='error_memory'; result_data['message']=f"{algo_display_text}: Memory Error!"; print(f"MemError {algo_func_name}: {me}")
                except Exception as e: t_elapsed=time.perf_counter()-t_start; result_data['time']=t_elapsed; result_data['status']='error_runtime'; result_data['message']=f"{algo_display_text}: Runtime Error!"; print(f"RuntimeErr {algo_func_name}: {e}"); traceback.print_exc()
            else: result_data['message'] = f"Error: Func '{algo_func_name}' not found!"; result_data['status'] = 'error_missing_func'
        except Exception as e: result_data['message'] = f"Thread setup error: {e}"; result_data['status'] = 'error_setup'; print(f"ThreadSetupErr {algo_func_name}: {e}"); traceback.print_exc()
        try: q.put(result_data)
        except Exception as qe: print(f"CRITICAL QUEUE PUT ERROR: {qe}")

    def _check_result_queue(self):
        try:
            result = self.result_queue.get_nowait()
            print(f"Result received: {result['status']} - {result['algo_text']}")
            if result['algo_name'] == 'benchmark':
                 if result['status'] == 'benchmark_complete': self._process_benchmark_result(result)
                 elif result['status'].startswith('error'): self.set_message(f"Benchmark Error: {result.get('message', 'Unknown')}", MSG_ERROR_COLOR); self.app_state = "FAILED"; self.update_button_states()
            else: self._process_solver_result(result)
        except queue.Empty: pass
        except Exception as e: print(f"Error processing queue: {e}"); traceback.print_exc(); self.set_message("Result processing error!", MSG_ERROR_COLOR); self.app_state = "FAILED"; self.update_button_states()

    # Process solver result remains the same as previous version
    def _process_solver_result(self, result):
        algo_text = result['algo_text']; algo_config = self.buttons_config.get(algo_text, {}); algo_type = algo_config.get('type', 'unknown')
        status = result['status']; message = result['message']; path_or_state = result['path']; t_elapsed = result['time']
        result_lines_display = []
        start_visualization = False
        if status.startswith('error'):
            print(f"Error reported for {algo_text}: {message}"); self.set_message(message, MSG_ERROR_COLOR); self.app_state = "FAILED"
            result_lines_display = [f"Status {algo_text}: Error", message, f"(Time: {t_elapsed:.3f}s)"]
        elif status == 'failure':
             res_msg = f"{algo_text}: Failed/Stopped ({t_elapsed:.3f}s)"; print(res_msg); self.set_message(res_msg, MSG_WARN_COLOR); self.app_state = "FAILED"
             result_lines_display = [f"Status {algo_text}: Failed", message, f"(Time: {t_elapsed:.3f}s)"]
             if algo_type == 'generate_explore' and path_or_state: # Show backtrack exploration on fail
                  self.solution_path = path_or_state; steps_exp = len(self.solution_path)
                  self.set_message(f"{algo_text}: Stopped ({steps_exp} states, {t_elapsed:.3f}s)", MSG_WARN_COLOR)
                  result_lines_display = [f"{algo_text} Stopped ({t_elapsed:.3f}s):", f"Explored {steps_exp} states."]
                  self.app_state = "VISUALIZING"; self.final_state_after_vis = "FAILED"; start_visualization = True
        elif status.startswith('success'):
            try:
                if not self.puzzle_instance: raise RuntimeError("Puzzle instance missing!")
                self.solution_path = path_or_state
                is_goal = (status == 'success_goal')
                self.final_state_after_vis = "SOLVED" if is_goal else "FAILED" # Set expected final state
                if algo_type in ['path', 'path_if_found', 'beam_search']:
                    steps = len(self.solution_path) - 1
                    res_msg = f"{algo_text}: Solved! ({steps} steps, {t_elapsed:.3f}s)" if is_goal else f"{algo_text}: Finished ({steps} steps, H={self.puzzle_instance.heuristic(path_or_state[-1])}, {t_elapsed:.3f}s)"
                    print(res_msg); self.set_message(res_msg, MSG_SUCCESS_COLOR if is_goal else MSG_WARN_COLOR)
                    moves = get_solution_moves(self.solution_path) if is_goal else ["Path shown in animation"]
                    result_lines_display = [f"{algo_text} {'Solved' if is_goal else 'Finished'} ({t_elapsed:.3f}s):", f"Steps: {steps}"] + (["--- Moves ---"] + [f"{i+1}. {m}" for i, m in enumerate(moves)] if is_goal else [])
                    self.app_state = "VISUALIZING"; start_visualization = True
                elif algo_type in ['explore_path', 'generate_explore']: # CSP, Backtrack
                    steps_exp = len(self.solution_path)
                    path_len = len(path_or_state) -1 if algo_type == 'explore_path' and is_goal else steps_exp
                    status_desc = f"Solved (Path {path_len})" if is_goal else f"Explored ({steps_exp} states)"
                    res_msg = f"{algo_text}: {status_desc}! ({t_elapsed:.3f}s)"
                    print(res_msg); self.set_message(res_msg, MSG_SUCCESS_COLOR if is_goal else MSG_WARN_COLOR)
                    result_lines_display = [f"{algo_text} {status_desc} ({t_elapsed:.3f}s):", f"{'Found path' if is_goal else 'Explored'} {steps_exp} states."]
                    self.app_state = "VISUALIZING"; start_visualization = True
                elif algo_type == 'local': # HC, SA
                    steps_local = len(self.solution_path) - 1; final_state = path_or_state[-1]; final_h = self.puzzle_instance.heuristic(final_state)
                    status_desc = "Solved" if is_goal else f"Finished (H={final_h})"
                    res_msg = f"{algo_text}: {status_desc} ({steps_local} steps, {t_elapsed:.3f}s)"
                    print(res_msg); self.set_message(res_msg, MSG_SUCCESS_COLOR if is_goal else MSG_WARN_COLOR)
                    result_lines_display = [f"{algo_text} Result ({t_elapsed:.3f}s):", status_desc, f"Steps: {steps_local}"]
                    self.app_state = "VISUALIZING"; start_visualization = True
                elif algo_type == 'state_only': # GA
                     final_state_ga = path_or_state[0]; final_h_ga = self.puzzle_instance.heuristic(final_state_ga)
                     status_desc = "Solved (Genetic)" if is_goal else f"Finished (Best H={final_h_ga})"
                     res_msg = f"{algo_text}: {status_desc} ({t_elapsed:.3f}s)"
                     print(res_msg); self.set_message(res_msg, MSG_SUCCESS_COLOR if is_goal else MSG_WARN_COLOR)
                     self.update_grid_display(final_state_ga); self.app_state = "SOLVED" if is_goal else "FAILED" # No vis for GA
                     result_lines_display = [f"{algo_text} Result ({t_elapsed:.3f}s):", status_desc]
                else: raise ValueError(f"Unhandled type '{algo_type}' for success.")
            except Exception as proc_err:
                 print(f"--- Error processing success result {algo_text} ---"); traceback.print_exc()
                 self.set_message(f"Error processing {algo_text} result!", MSG_ERROR_COLOR); self.app_state = "FAILED"
                 result_lines_display = [f"Status {algo_text}: Processing Error", str(proc_err)]
        # Update UI
        self.set_results(result_lines_display)
        if start_visualization: self.vis_step_index = 0; self.last_vis_time = pygame.time.get_ticks() # Start timer only if visualizing
        if self.app_state != "VISUALIZING": self.update_button_states()

    def _update_visualization(self):
        if self.app_state != "VISUALIZING" or not self.solution_path: return
        now = pygame.time.get_ticks(); current_delay = self.vis_delay
        algo_type = self.buttons_config.get(self.selected_algorithm_text, {}).get('type', 'path')
        if algo_type in ['explore_path', 'generate_explore']: current_delay = max(10, self.vis_delay // 3)
        if now - self.last_vis_time > current_delay:
            self.last_vis_time = now
            if self.vis_step_index < len(self.solution_path):
                current_step_state = self.solution_path[self.vis_step_index]
                if is_valid_state(current_step_state):
                    self.update_grid_display(current_step_state)
                    steps_total = len(self.solution_path); prefix = "Step"; step_disp = self.vis_step_index; total_disp = steps_total - 1
                    if algo_type in ['explore_path', 'generate_explore']: prefix = "Explore"; step_disp += 1; total_disp = steps_total
                    elif algo_type == 'local': prefix = "Local Step"; step_disp +=1; total_disp = steps_total
                    if algo_type in ['path', 'path_if_found', 'beam_search'] and step_disp == 0: self.set_message(f"{self.selected_algorithm_text} Start...", MSG_INFO_COLOR)
                    else: self.set_message(f"{prefix}: {step_disp}/{total_disp}", MSG_INFO_COLOR)
                    self.vis_step_index += 1
                else:
                    print(f"Vis Error: Invalid state step {self.vis_step_index}."); self.set_message("Visualization Error!", MSG_ERROR_COLOR)
                    self.app_state = self.final_state_after_vis or "FAILED"; self.update_button_states(); self.solution_path = None;
            else: # Visualization finished
                self.app_state = self.final_state_after_vis or "FAILED"
                final_msg = f"{self.selected_algorithm_text} Finished!"; final_color = MSG_SUCCESS_COLOR if self.app_state == "SOLVED" else MSG_WARN_COLOR
                time_str_vis = ""
                try: # Extract time from results
                     if self.results_lines:
                         for line in self.results_lines[:2]:
                             if "(" in line and ("s):" in line or "s):" in line): time_part = line.split('(')[-1].split('s)')[0]; time_str_vis = f" ({time_part}s)"; break
                except Exception: pass
                self.set_message(final_msg + time_str_vis, final_color); self.update_button_states()
                self.solution_path = None; self.vis_step_index = 0; self.final_state_after_vis = None

    # --- Benchmark ---
    def run_benchmark_threaded(self):
        print("\n--- Preparing Benchmark ---")
        if not self.user_start_state or not is_solvable(self.user_start_state): self.set_message("Cannot Benchmark: Board not ready/unsolvable.", MSG_ERROR_COLOR); return
        self.app_state = "BENCHMARKING"; self.update_button_states()
        self.set_message("Running Benchmark...", MSG_INFO_COLOR); self.set_results(["Benchmark Results:", "Running algorithms..."])
        self.draw(); pygame.display.flip() # Show message
        start_state_copy = [r[:] for r in self.user_start_state]
        thread = threading.Thread(target=self._benchmark_thread_func, args=(start_state_copy, self.result_queue), daemon=True)
        thread.start()

    # Benchmark thread function remains the same as previous version
    def _benchmark_thread_func(self, start_state_input, q):
        benchmark_results_list = []; total_time = 0; print("--- Starting Benchmark Run ---")
        for algo_func_bench in self.pathfinding_algos_for_benchmark:
            if algo_func_bench in self.algo_name_map:
                algo_txt_bench = self.algo_name_map[algo_func_bench]; print(f"Benchmarking: {algo_txt_bench}...")
                path_b, time_b, status_b, steps_b = None, 0, "Error", -1
                t_start_b = time.perf_counter()
                try:
                    puzzle_b = Puzzle(start_state_input)
                    if algo_func_bench in ['iddfs', 'and_or_search']: puzzle_b.iddfs_max_depth = int(self.sliders['max_depth']['value'])
                    # Apply other params if needed
                    solver_b = getattr(puzzle_b, algo_func_bench, None)
                    if solver_b:
                        path_b = solver_b(); time_b = time.perf_counter() - t_start_b; total_time += time_b
                        if path_b and isinstance(path_b, list) and len(path_b) > 0:
                            last_state = path_b[-1]; is_goal = puzzle_b.is_goal(last_state)
                            if is_goal:
                                status_b = "Solved"; algo_type_b = self.buttons_config[algo_txt_bench]['type']
                                if algo_type_b in ['path', 'path_if_found', 'beam_search']: steps_b = len(path_b)-1
                                elif algo_type_b in ['explore_path', 'generate_explore']: steps_b = len(path_b)
                            else: status_b = "Finished"; steps_b = len(path_b) # Steps taken or explored
                        else: status_b = "Failed/Stopped"
                    else: status_b = "Not Found"; time_b = time.perf_counter() - t_start_b; total_time += time_b
                except MemoryError: status_b = "Memory Error"; time_b = time.perf_counter() - t_start_b; total_time += time_b
                except Exception as e: status_b = "Runtime Error"; time_b = time.perf_counter() - t_start_b; total_time += time_b; print(f"Bench Err {algo_txt_bench}: {e}")
                print(f"... {algo_txt_bench}: {status_b}, Steps/States: {steps_b if steps_b != -1 else 'N/A'}, Time: {time_b:.4f}s")
                benchmark_results_list.append({'name': algo_txt_bench, 'time': time_b, 'steps': steps_b, 'status': status_b})
            else: print(f"Skipping unknown func: {algo_func_bench}")
        print(f"--- Benchmark Finished. Total Time: {total_time:.3f}s ---")
        benchmark_result_data = {'status': 'benchmark_complete', 'message': f"Benchmark complete ({total_time:.2f}s)", 'results': benchmark_results_list, 'total_time': total_time, 'algo_name': 'benchmark', 'algo_text': 'Benchmark'}
        try: q.put(benchmark_result_data)
        except Exception as qe: print(f"CRITICAL QUEUE PUT ERROR (Benchmark): {qe}")

    # Process benchmark result remains the same as previous version
    def _process_benchmark_result(self, result):
        print("Processing Benchmark Result"); all_res = result.get('results', []); total_t = result.get('total_time', 0)
        results_display = ["Benchmark Results:", "---------------------------------"]; all_res.sort(key=lambda x: x['name'])
        max_name_len = max(max(len(r['name']) for r in all_res), 12) if all_res else 12
        for res in all_res:
            t_str = f"{res['time']:.3f}s" if res['time'] > 0.0001 else "-"; name_s = res['name'].ljust(max_name_len)
            status_s = res['status']; steps_s = f"{res['steps']}" if res['steps'] != -1 else "N/A"
            steps_label = "Steps"; algo_type = self.buttons_config.get(res['name'], {}).get('type', '')
            if algo_type in ['explore_path', 'generate_explore'] : steps_label = "States"
            line = f"{name_s}: {status_s:<18} ({steps_s} {steps_label}), {t_str:>8}"
            results_display.append(line)
        results_display.append("---------------------------------")
        solved_path_res = [r for r in all_res if "Solved" in r['status'] and r['steps'] != -1 and self.buttons_config.get(r['name'], {}).get('type', '') in ['path', 'path_if_found', 'beam_search']]
        sug = "Suggestion: No pathfinding solution completed."
        if solved_path_res:
            try:
                min_s = min((r['steps'] for r in solved_path_res), default=-1)
                if min_s != -1:
                    optimal_solvers = sorted([r for r in solved_path_res if r['steps'] == min_s], key=lambda x: x['time'])
                    if optimal_solvers: best = optimal_solvers[0]; sug = f"Suggestion: {best['name']} optimal ({best['steps']} steps, {best['time']:.3f}s)"
                    else: sug = "Suggestion: Error calculating optimal."
                else: # Fallback: Find fastest solver
                    fastest = min(solved_path_res, key=lambda x: x['time'])
                    sug = f"Suggestion: Fastest: {fastest['name']} ({fastest['steps']} steps, {fastest['time']:.3f}s)"
            except Exception as e: print(f"Suggest Error: {e}"); sug = "Suggestion: Error generating suggestion."
        results_display.append(sug)
        self.set_results(results_display); self.set_message(f"Benchmark complete ({total_t:.2f}s). See Results.", MSG_DEFAULT_COLOR)
        if self.user_start_state and is_solvable(self.user_start_state): self.app_state = "READY"
        elif self.app_state == "BENCHMARKING": self.app_state = "FAILED"
        self.update_button_states()

    # --- Main Update and Draw ---
    def update(self):
        """Update game state, animations, and check threads."""
        if self.app_state == "PLACING_RANDOM": self._update_placement_animation()
        elif self.app_state == "VISUALIZING": self._update_visualization()
        self._check_result_queue() # Check threads always

    def draw(self):
        """Draw all elements onto the screen."""
        self.screen.fill(LIGHT_GREY)
        # Draw UI Area backgrounds (optional)
        pygame.draw.rect(self.screen, WHITE, self.controls_rect.inflate(10,10), border_radius=5)
        pygame.draw.rect(self.screen, WHITE, self.results_rect.inflate(4,4), border_radius=5)
        pygame.draw.rect(self.screen, LIGHT_GREY, self.bottom_panel_rect) # Background for sliders/notes
        pygame.draw.rect(self.screen, DARK_GREY, self.bottom_panel_rect, 1, border_radius=3) # Border for bottom panel

        self.draw_grid()
        self.draw_buttons()
        self.draw_message()
        self.draw_results()
        self.draw_sliders()
        self.draw_notes() # Draw notes in the bottom panel
        pygame.display.flip()

    # --- Main Game Loop ---
    def run(self):
        """Main application loop."""
        while self.running:
            self.handle_input()
            self.update()
            self.draw()
            self.clock.tick(60) # Limit FPS
        print("Exiting Pygame application...")
        pygame.quit()


# --- Main Execution ---
if __name__ == "__main__":
    try:
        app = PygamePuzzleApp()
        app.run()
    except Exception as main_exception:
        print("\n--- UNHANDLED EXCEPTION IN MAIN APPLICATION ---")
        traceback.print_exc()
        pygame.quit()
        sys.exit(1)