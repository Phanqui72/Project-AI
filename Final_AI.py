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
import re # For regex in visualization time extraction

# --- Pygame Initialization and Basic Setup ---
try:
    pygame.init()
    if not pygame.font:
        print("Pygame Font module not initialized!")
        pygame.font.init()
        if not pygame.font:
             raise RuntimeError("Pygame Font module failed to initialize.")
    print("Pygame initialized successfully.")
except Exception as e:
    print(f"Fatal Error initializing Pygame: {e}")
    sys.exit(1)

# --- Screen Dimensions and Layout ---
GRID_SIZE = 450
TILE_SIZE = GRID_SIZE // 3
PANEL_PADDING = 15
CONTROLS_WIDTH = 220
RESULTS_WIDTH = 500
BOTTOM_MARGIN = 60

WIDTH = GRID_SIZE + CONTROLS_WIDTH + RESULTS_WIDTH + PANEL_PADDING * 4
HEIGHT = GRID_SIZE + PANEL_PADDING * 2 + BOTTOM_MARGIN

GRID_X = PANEL_PADDING
GRID_Y = PANEL_PADDING
CONTROLS_X = GRID_X + GRID_SIZE + PANEL_PADDING
CONTROLS_Y = GRID_Y
RESULTS_X = CONTROLS_X + CONTROLS_WIDTH + PANEL_PADDING
RESULTS_Y = GRID_Y
RESULTS_HEIGHT = GRID_SIZE

BOTTOM_PANEL_Y = GRID_Y + GRID_SIZE + PANEL_PADDING
BOTTOM_PANEL_HEIGHT = BOTTOM_MARGIN
BOTTOM_PANEL_WIDTH = CONTROLS_WIDTH + RESULTS_WIDTH + PANEL_PADDING
BOTTOM_PANEL_X = CONTROLS_X

# --- Colors ---
WHITE = (255, 255, 255); BLACK = (0, 0, 0); GREY = (200, 200, 200)
DARK_GREY = (100, 100, 100); LIGHT_GREY = (240, 240, 240)
RED = (200, 0, 0); GREEN = (0, 200, 0); BLUE = (70, 130, 180)
DARK_BLUE = (0, 0, 139); ORANGE = (255, 165, 0); PURPLE = (128, 0, 128)
DARK_RED = (150, 50, 50); DARK_GREEN = (0, 100, 0)
NOTE_COLOR = (80, 80, 80)
TILE_BORDER = (64, 64, 64)
TILE_EMPTY_BG = (220, 220, 220)
BUTTON_COLOR = (0, 120, 215)
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
    (255, 99, 71), (255, 127, 80), (255, 140, 0), (255, 160, 122),
    (144, 238, 144), (143, 188, 143), (60, 179, 113), (46, 139, 87),
    (173, 216, 230)
]

# --- Fonts ---
try:
    FONT_PATH = pygame.font.match_font(['segoeui', 'calibri', 'arial', 'sans'])
    if FONT_PATH is None: FONT_PATH = pygame.font.get_default_font()
    TILE_FONT = pygame.font.Font(FONT_PATH, 50)
    BUTTON_FONT = pygame.font.Font(FONT_PATH, 16)
    MSG_FONT = pygame.font.Font(FONT_PATH, 17)
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

# --- Trạng thái đích và Giới hạn ---
GOAL_STATE = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
GOAL_STATE_TUPLE = tuple(map(tuple, GOAL_STATE))
DEFAULT_MAX_DEPTH = 15; MAX_BFS_STATES = 500000; MAX_DFS_POPS = 750000
MAX_GREEDY_EXPANSIONS = 500000; MAX_HILL_CLIMBING_STEPS = 5000
MAX_SA_ITERATIONS = 30000; MAX_BEAM_STEPS = 1000; MAX_GA_GENERATIONS = 300
MAX_IDA_STAR_NODES = 2000000

# --- Helper Functions ---
def state_to_tuple(state):
    return tuple(map(tuple, state))

def is_valid_state(state):
    try:
        if isinstance(state, (list, tuple)) and len(state) == 3 and all(len(row) == 3 for row in state):
             if isinstance(state[0], tuple): flat = sum(state, ())
             else: flat = sum(state, [])
             return len(flat) == 9 and sorted(flat) == list(range(9))
        else: return False
    except (TypeError, ValueError): return False

def get_inversions(flat_state):
    count = 0; size = len(flat_state)
    for i in range(size):
        for j in range(i + 1, size):
            val_i, val_j = flat_state[i], flat_state[j]
            if isinstance(val_i, int) and isinstance(val_j, int) and \
               val_i != 0 and val_j != 0 and val_i > val_j:
                count += 1
    return count

def is_solvable(state, goal_state_ref=GOAL_STATE_TUPLE):
    if not is_valid_state(state): return False
    standard_goal_flat = list(range(9))
    target_inversions_parity = get_inversions(standard_goal_flat) % 2
    if isinstance(state, tuple): state_flat = sum(state, ())
    else: state_flat = sum(state, [])
    state_inversions = get_inversions(state_flat)
    return state_inversions % 2 == target_inversions_parity

def generate_random_solvable_state(goal_state_ref=GOAL_STATE_TUPLE):
    attempts = 0; max_attempts = 1000
    standard_goal_flat = list(range(9))
    target_inversions_parity = get_inversions(standard_goal_flat) % 2
    while attempts < max_attempts:
        flat = list(range(9)); random.shuffle(flat);
        current_inversions_parity = get_inversions(flat) % 2
        if current_inversions_parity == target_inversions_parity:
             state = [flat[i:i+3] for i in range(0, 9, 3)]
             if is_valid_state(state): return state
        attempts += 1
    print(f"Error: Could not generate a solvable state after {max_attempts} attempts.")
    return [list(row) for row in goal_state_ref]

def get_solution_moves(path):
    if not path or len(path) < 2: return ["No moves in path."]
    moves = [];
    def find_blank(state):
         for r, row in enumerate(state):
              for c, val in enumerate(row):
                   if val == 0: return r, c
         raise ValueError("Invalid state: Blank not found.")
    for i in range(len(path) - 1):
        s1, s2 = path[i], path[i+1]
        try:
            if not is_valid_state(s1) or not is_valid_state(s2):
                moves.append("Invalid state in path."); continue
            r1, c1 = find_blank(s1); r2, c2 = find_blank(s2)
            moved_tile_value = s2[r1][c1]
            move = f"Tile {moved_tile_value} moves "
            if r2 < r1: move += "Down"
            elif r2 > r1: move += "Up"
            elif c2 < c1: move += "Right"
            elif c2 > c1: move += "Left"
            else: move += "Error? (No move)"
            moves.append(move)
        except (ValueError, IndexError, TypeError) as e:
            moves.append(f"Error determining move ({e})"); traceback.print_exc()
            continue
    return moves

# --- Lớp PuzzleState (CẦN THIẾT CHO Q-LEARNING) ---
# Giả định một PuzzleState cơ bản. Nếu bạn có một định nghĩa chi tiết hơn,
# hãy đảm bảo nó có các thuộc tính/phương thức mà q_learning sử dụng.
class PuzzleState:
    def __init__(self, board, move=None, parent=None, puzzle_instance=None):
        self.board = [list(row) for row in board] # Đảm bảo là list of lists có thể thay đổi
        self.move = move  # Hành động dẫn đến trạng thái này (ví dụ: "UP", "DOWN")
        self.parent = parent
        self.puzzle = puzzle_instance # Tham chiếu đến đối tượng Puzzle để gọi get_neighbors

        # Thuộc tính để Q-learning sử dụng nếu cần, ví dụ:
        # self.g = 0
        # self.h = 0
        # self.f = 0

    def __str__(self):
        return str(self.board)

    def __eq__(self, other):
        if other is None or not isinstance(other, PuzzleState):
            return False
        return self.board == other.board

    def __hash__(self):
        return hash(tuple(map(tuple, self.board)))

    def get_children(self):
        """
        Trả về list các PuzzleState con.
        Mỗi PuzzleState con phải có thuộc tính 'move' được đặt.
        """
        if self.puzzle is None:
            # Điều này không nên xảy ra nếu PuzzleState được tạo đúng cách
            print("Lỗi: PuzzleState không có tham chiếu đến Puzzle instance.")
            return []

        children = []
        blank_r, blank_c = -1, -1
        try:
            blank_r, blank_c = self.puzzle.get_blank_position(self.board)
        except ValueError:
            return [] # Không tìm thấy ô trống

        # Định nghĩa các nước đi và tên tương ứng
        # (dr, dc, action_name)
        possible_moves = [
            (-1, 0, "UP"),    # Ô trống đi lên (số di chuyển xuống)
            (1, 0, "DOWN"),  # Ô trống đi xuống (số di chuyển lên)
            (0, -1, "LEFT"),  # Ô trống đi trái (số di chuyển sang phải)
            (0, 1, "RIGHT")   # Ô trống đi phải (số di chuyển sang trái)
        ]
        # Lưu ý: Tên hành động "UP", "DOWN", "LEFT", "RIGHT" này
        # mô tả hướng di chuyển của ô trống.
        # Nếu 'move' trong q_learning của bạn mong đợi tên của ô được di chuyển,
        # bạn cần điều chỉnh logic này.
        # Hàm q_learning gốc không nói rõ 'move' là gì, nhưng get_solution_moves
        # lại mô tả "Tile X moves Direction".
        # Để nhất quán, 'move' ở đây sẽ là hướng ô trống di chuyển.

        for dr, dc, action_name in possible_moves:
            new_r, new_c = blank_r + dr, blank_c + dc
            if 0 <= new_r < self.puzzle.rows and 0 <= new_c < self.puzzle.cols:
                new_board = [row[:] for row in self.board]
                # Di chuyển ô vào vị trí ô trống
                moved_tile = new_board[new_r][new_c]
                new_board[blank_r][blank_c], new_board[new_r][new_c] = new_board[new_r][new_c], new_board[blank_r][blank_c]

                # Quyết định xem 'move' là gì. Hàm q_learning dùng child.move.
                # Giả sử 'move' là hành động (ví dụ: "UP")
                # Nếu bạn muốn 'move' là ô nào đã di chuyển, bạn cần logic khác.
                # Ví dụ: `move_description = f"Tile {moved_tile} to ({blank_r},{blank_c})"`
                # Hoặc chỉ đơn giản là action_name (hướng ô trống di chuyển)
                children.append(PuzzleState(new_board, move=action_name, parent=self, puzzle_instance=self.puzzle))
        return children


# --- Lớp Puzzle ---
class Puzzle:
    def __init__(self, start, goal=GOAL_STATE):
        start_list = [list(row) for row in start]
        if not is_valid_state(start_list):
            raise ValueError("Invalid start state provided to Puzzle.")
        self.start = start_list
        self.goal = [list(row) for row in goal]
        self.goal_tuple = state_to_tuple(self.goal)
        self.rows = 3; self.cols = 3
        self._goal_pos_cache = self._build_goal_pos_cache()
        self.iddfs_max_depth = DEFAULT_MAX_DEPTH
        self.sa_initial_temp = 1000
        self.sa_cooling_rate = 0.997
        self.sa_min_temp = 0.1
        self.beam_width = 5
        self.ga_pop_size = 60
        self.ga_mutation_rate = 0.15
        self.ga_elite_size = 5
        self.ga_tournament_k = 3
        

    def _build_goal_pos_cache(self):
        cache = {};
        for r in range(self.rows):
            for c in range(self.cols):
                if self.goal[r][c] != 0: cache[self.goal[r][c]] = (r, c)
        return cache

    def get_blank_position(self, state):
        for r in range(self.rows):
            for c in range(self.cols):
                if state[r][c] == 0: return r, c
        raise ValueError("Invalid state: Blank (0) not found.")

    def is_goal(self, state):
        return state_to_tuple(state) == self.goal_tuple

    def get_neighbors(self, state):
        neighbors = [];
        try: r, c = self.get_blank_position(state)
        except ValueError: return neighbors
        moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        current_state_list = [list(row) for row in state]
        for dr, dc in moves:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                new_state = [row[:] for row in current_state_list]
                new_state[r][c], new_state[nr][nc] = new_state[nr][nc], new_state[r][c]
                neighbors.append(new_state)
        return neighbors

    def heuristic(self, state):
        distance = 0
        state_tuple = state if isinstance(state, tuple) else state_to_tuple(state)
        for r in range(self.rows):
            for c in range(self.cols):
                val = state_tuple[r][c]
                if val != 0:
                    if val in self._goal_pos_cache:
                        goal_r, goal_c = self._goal_pos_cache[val]
                        distance += abs(r - goal_r) + abs(c - goal_c)
                    else: return float('inf')
        return distance

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
        pq = [(0, start_tuple, [self.start])]
        visited = {start_tuple: 0}; count = 0; max_expansions = MAX_BFS_STATES
        while pq and count < max_expansions:
            cost, current_tuple, path = heapq.heappop(pq)
            if cost > visited[current_tuple]: continue
            count += 1; current_state = path[-1]
            if self.is_goal(current_state): print(f"UCS: Solved! Cost={cost}, Expanded={count}"); return path
            for neighbor_state in self.get_neighbors(current_state):
                neighbor_tuple = state_to_tuple(neighbor_state); new_cost = cost + 1
                if neighbor_tuple not in visited or new_cost < visited[neighbor_tuple]:
                    visited[neighbor_tuple] = new_cost
                    heapq.heappush(pq, (new_cost, neighbor_tuple, path + [neighbor_state]))
        status = 'Limit reached' if count >= max_expansions else 'Queue empty'
        print(f"UCS: Failed ({status})"); return None

    def dfs(self):
        print("DFS: Starting..."); st=[(self.start,[self.start])]; v={state_to_tuple(self.start)}; c=0
        dfs_depth_limit = self.iddfs_max_depth + 15
        while st and c<MAX_DFS_POPS:
            s,p=st.pop(); c+=1;
            if self.is_goal(s): print(f"DFS: Solved ({c} pops)"); return p
            if len(p) > dfs_depth_limit: continue
            for n in reversed(self.get_neighbors(s)):
                nt=state_to_tuple(n);
                if nt not in v: v.add(nt); st.append((n,p+[n]))
        status = 'Limit reached' if c>=MAX_DFS_POPS else 'Stack empty'; print(f"DFS: Failed/Limit ({status})"); return None

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
            if result_path: print(f"IDDFS: Solved! (Depth={depth}, Nodes ~{nodes_total})"); return result_path
        print(f"IDDFS: Failed (Max Depth {self.iddfs_max_depth} reached, Nodes ~{nodes_total})"); return None

    def greedy(self):
        print("Greedy: Starting..."); s_t=state_to_tuple(self.start);
        pq=[(self.heuristic(self.start), s_t, [self.start])];
        visited={s_t}; count=0
        while pq and count<MAX_GREEDY_EXPANSIONS:
            _, current_tuple, path = heapq.heappop(pq); count+=1
            current_state = path[-1]
            if self.is_goal(current_state): print(f"Greedy: Solved ({count} expansions)"); return path
            for neighbor_state in self.get_neighbors(current_state):
                neighbor_tuple=state_to_tuple(neighbor_state);
                if neighbor_tuple not in visited:
                    visited.add(neighbor_tuple); h = self.heuristic(neighbor_state)
                    if h != float('inf'): heapq.heappush(pq,(h, neighbor_tuple, path+[neighbor_state]))
        status = 'Limit reached' if count>=MAX_GREEDY_EXPANSIONS else 'Queue empty'; print(f"Greedy: Failed/Limit ({status})"); return None

    def a_star(self):
        print("A*: Starting..."); s_t=state_to_tuple(self.start);
        open_list = [(self.heuristic(self.start) + 0, 0, self.start, s_t)]
        g_costs = {s_t: 0}; came_from = {s_t: (None, None)}
        closed_set = set(); count = 0; max_expansions = MAX_IDA_STAR_NODES
        while open_list and count < max_expansions:
            f, g, current_state, current_tuple = heapq.heappop(open_list); count += 1
            if g > g_costs.get(current_tuple, float('inf')): continue
            if current_tuple in closed_set: continue
            closed_set.add(current_tuple)
            if current_tuple == self.goal_tuple:
                path = []; curr_t = current_tuple; final_state_in_pq = current_state
                path.append(final_state_in_pq)
                pred_t, pred_s = came_from.get(curr_t, (None, None))
                while pred_t is not None:
                    if pred_s is None: print("A* Path Error: Predecessor state is None!"); break
                    path.append(pred_s); curr_t = pred_t
                    pred_t, pred_s = came_from.get(curr_t, (None, None))
                if not path or state_to_tuple(path[-1]) != s_t:
                     if state_to_tuple(self.start) == s_t: path.append(self.start)
                path.reverse()
                print(f"A*: Solved! (Expanded={count}, Cost={g})"); return path
            for neighbor_state in self.get_neighbors(current_state):
                neighbor_tuple = state_to_tuple(neighbor_state);
                if neighbor_tuple in closed_set: continue
                tentative_g = g + 1
                if tentative_g < g_costs.get(neighbor_tuple, float('inf')):
                    g_costs[neighbor_tuple] = tentative_g
                    came_from[neighbor_tuple] = (current_tuple, current_state)
                    h = self.heuristic(neighbor_state);
                    if h == float('inf'): continue
                    f_new = tentative_g + h;
                    heapq.heappush(open_list,(f_new, tentative_g, neighbor_state, neighbor_tuple))
        status = 'Limit reached' if count >= max_expansions else 'Queue empty'
        print(f"A*: Failed ({status})"); return None

    def ida_star(self):
        print("IDA*: Starting..."); start_tuple=state_to_tuple(self.start);
        bound = self.heuristic(self.start); path = [self.start]; nodes_expanded = 0
        def search(current_path, g_cost, current_bound):
            nonlocal nodes_expanded; nodes_expanded += 1
            if nodes_expanded > MAX_IDA_STAR_NODES: raise MemoryError(f"IDA* Node limit ({MAX_IDA_STAR_NODES}) reached")
            current_state = current_path[-1]; h_cost = self.heuristic(current_state)
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
                if found: print(f"IDA*: Solved! (Bound={bound}, Total Nodes ~{nodes_expanded})"); return result
                elif result == float('inf'): print("IDA*: Failed (No solution possible)"); return None
                else: bound = result if result > bound else bound + 1
            except MemoryError as me: print(f"IDA*: Failed ({me})"); return None
            if bound > 100: print(f"IDA*: Stopping (Bound {bound} > 100)"); return None

    def beam_search(self, beam_width=None):
        bw = beam_width if beam_width is not None else self.beam_width
        print(f"Beam Search: Starting (Width={bw})...");
        start_tuple=state_to_tuple(self.start);
        beam = [(self.heuristic(self.start), self.start, [self.start])];
        visited = {start_tuple}; step = 0; best_goal_path = None
        while beam and step < MAX_BEAM_STEPS:
            step += 1; candidates = []; candidates_tuples_this_step = set()
            for h, current_state, path_so_far in beam: # Renamed path to path_so_far
                if self.is_goal(current_state):
                    if best_goal_path is None or len(path_so_far) < len(best_goal_path):
                        best_goal_path = path_so_far
                for neighbor_state in self.get_neighbors(current_state):
                    neighbor_tuple = state_to_tuple(neighbor_state);
                    if (neighbor_tuple not in visited or self.is_goal(neighbor_state)) and \
                       neighbor_tuple not in candidates_tuples_this_step:
                         neighbor_h = self.heuristic(neighbor_state)
                         if neighbor_h != float('inf'):
                            candidates.append((neighbor_h, neighbor_state, path_so_far + [neighbor_state]));
                            visited.add(neighbor_tuple); candidates_tuples_this_step.add(neighbor_tuple)
            if not candidates:
                 if best_goal_path: print(f"Beam Search: Solved (Stuck step {step})"); return best_goal_path
                 else: print(f"Beam Search: Failed (Stuck step {step})"); return None
            candidates.sort(key=lambda x: x[0]); beam = candidates[:bw]
        if best_goal_path: print(f"Beam Search: Solved (Max steps {step}, best len {len(best_goal_path)-1})"); return best_goal_path
        else: print(f"Beam Search: Failed/Limit (Max steps {step})"); return None

    def backtracking(self): # Optimized version
        print("Backtrack: Starting...") # Giữ lại thông báo gốc để nhất quán
        start_tuple = state_to_tuple(self.start)

        # Stack lưu trữ: (current_state_object, current_depth)
        # current_state_object là một list các list biểu diễn trạng thái puzzle
        # current_depth là độ sâu của trạng thái trong cây tìm kiếm
        stack = [(self.start, 0)] 
        
        # visited_tuples lưu trữ các tuple của trạng thái đã được thêm vào stack hoặc đã xử lý
        # Điều này quan trọng để tránh chu trình và công việc dư thừa.
        visited_tuples = {start_tuple}

        # came_from lưu trữ {child_state_tuple: predecessor_state_object}
        # Điều này cho phép tái tạo đường đi của các đối tượng state *objects* nếu tìm thấy giải pháp.
        # predecessor_state_object được lưu trữ để tránh tạo lại đối tượng hoặc chuyển đổi lại tuple.
        came_from = {start_tuple: None} # Trạng thái gốc không có predecessor

        # exploration_sequence_objects lưu trữ các state_objects theo thứ tự chúng được pop ra khỏi stack.
        # Chuỗi này được trả về nếu không tìm thấy giải pháp trong giới hạn.
        exploration_sequence_objects = [] 
        
        pops_count = 0
        max_pops_limit = MAX_DFS_POPS # Sử dụng giới hạn hiện có cho số lần pop

        solution_path_if_found = None

        while stack and pops_count < max_pops_limit:
            current_state_obj, current_depth = stack.pop() # LIFO cho hành vi DFS
            pops_count += 1
            exploration_sequence_objects.append(current_state_obj)
            
            if self.is_goal(current_state_obj):
                # Đã tìm thấy trạng thái đích, tái tạo đường đi từ trạng thái bắt đầu đến current_state_obj
                path = []
                temp_state_obj_for_path = current_state_obj 
                
                # Duyệt ngược từ trạng thái đích sử dụng came_from
                while temp_state_obj_for_path is not None:
                    path.append(temp_state_obj_for_path)
                    # Chuyển đổi đối tượng hiện tại thành tuple để tra cứu cha của nó trong came_from
                    parent_lookup_key = state_to_tuple(temp_state_obj_for_path)
                    temp_state_obj_for_path = came_from.get(parent_lookup_key)
                
                path.reverse() # Đường đi được xây dựng từ đích về đầu, nên đảo ngược lại
                solution_path_if_found = path
                
                # Thông báo này giữ nguyên format so với bản gốc, sử dụng độ dài đường đi đã tái tạo
                print(f"Backtrack: Solved! ({pops_count} pops, path len {len(solution_path_if_found)-1})")
                break # Thoát vòng lặp while vì đã tìm thấy giải pháp

            # Thêm các trạng thái lân cận vào stack
            # Lặp theo thứ tự đảo ngược của get_neighbors() có nghĩa là
            # neighbor đầu tiên được trả về bởi get_neighbors() sẽ được xử lý trước (LIFO).
            for neighbor_obj in reversed(self.get_neighbors(current_state_obj)):
                neighbor_tuple = state_to_tuple(neighbor_obj)
                if neighbor_tuple not in visited_tuples:
                    visited_tuples.add(neighbor_tuple)
                    # Ghi nhận current_state_obj là predecessor của neighbor_obj
                    came_from[neighbor_tuple] = current_state_obj 
                    stack.append((neighbor_obj, current_depth + 1))
        
        if solution_path_if_found:
            # Nếu tìm thấy giải pháp, trả về đường đi đến trạng thái đích.
            # Điều này nhất quán với cách csp_solve (cũng là loại 'generate_explore') trả về.
            return solution_path_if_found
        else:
            # Nếu không tìm thấy giải pháp (do giới hạn hoặc stack rỗng),
            # trả về chuỗi tất cả các trạng thái đã khám phá.
            status = 'Limit reached' if pops_count >= max_pops_limit else 'Stack empty'
            print(f"Backtrack: Failed/Limit ({status}, {pops_count} pops explored)")
            return exploration_sequence_objects

    def csp_solve(self):
        print("CSP Solve (via Backtrack): Starting...")
        start_tuple = state_to_tuple(self.start)
        stack = [(self.start, [self.start])]
        visited = {start_tuple}; exploration_sequence = []; pops_count = 0; max_pops = MAX_DFS_POPS
        while stack and pops_count < max_pops:
            current_state, path_to_state = stack.pop(); pops_count += 1; # Renamed path
            exploration_sequence.append(current_state)
            if self.is_goal(current_state):
                 print(f"CSP Solve: Solved! ({pops_count} pops, path len {len(path_to_state)-1})")
                 return path_to_state # Return the path to goal
            for neighbor in reversed(self.get_neighbors(current_state)):
                neighbor_tuple = state_to_tuple(neighbor)
                if neighbor_tuple not in visited:
                    visited.add(neighbor_tuple); stack.append((neighbor, path_to_state + [neighbor]))
        status = 'Limit reached' if pops_count >= max_pops else 'Stack empty'
        print(f"CSP Solve: Failed/Limit ({status}, {pops_count} pops explored)")
        return exploration_sequence # Return full exploration if no solution

    def and_or_search(self):
        max_depth = self.iddfs_max_depth
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

    def _hill_climbing_base(self, find_next_state_func, name):
        print(f"{name}: Starting..."); current_state = [r[:] for r in self.start]
        path_taken = [[r[:] for r in current_state]]; visited_tuples = {state_to_tuple(current_state)}; steps = 0
        while not self.is_goal(current_state) and steps < MAX_HILL_CLIMBING_STEPS:
            steps += 1; current_heuristic = self.heuristic(current_state)
            if current_heuristic == float('inf'): print(f"{name}: Stuck H=inf step {steps}. Aborting."); return path_taken
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
            iterations += 1; neighbors = self.get_neighbors(current_state);
            if not neighbors: print(f"SA: Stuck iter {iterations}"); break
            next_state = random.choice(neighbors); next_h = self.heuristic(next_state);
            if next_h == float('inf'): continue
            delta_e = next_h - current_h; accept = False
            if delta_e < 0: accept = True
            else:
                 if temp > 1e-9:
                      try: accept_prob = math.exp(-delta_e / temp);
                      except OverflowError: accept_prob = 0
                      if random.random() < accept_prob: accept = True
                 else: accept = False
            if accept:
                current_state = next_state; current_h = next_h; current_tuple = state_to_tuple(current_state)
                if current_tuple != last_accepted_tuple:
                    path_taken.append([r[:] for r in current_state]); last_accepted_tuple = current_tuple
            temp *= cool_rate
        final_h = self.heuristic(current_state); goal_reached = " (Goal)" if self.is_goal(current_state) else ""
        reason = "Min Temp" if temp <= final_temp else f"Max Iter ({MAX_SA_ITERATIONS})"
        print(f"SA: Finished ({reason}). Iter={iterations}, Final T={temp:.2f}, Final H={final_h}{goal_reached}"); return path_taken

    def genetic_algorithm_solve(self, population_size=None, mutation_rate=None, elite_size=None, tournament_k=None):
        pop_size = population_size if population_size is not None else self.ga_pop_size
        mut_rate = mutation_rate if mutation_rate is not None else self.ga_mutation_rate
        elite_s = elite_size if elite_size is not None else self.ga_elite_size
        tourn_k = tournament_k if tournament_k is not None else self.ga_tournament_k
        print(f"GA: Starting (Pop={pop_size}, Mut={mut_rate:.2f}, Elite={elite_s}, TournK={tourn_k})...");
        
        visualization_frames = [] # <--- THÊM DÒNG NÀY

        def state_to_flat(state): return sum(state, [])
        def flat_to_state(flat_list): return [flat_list[i:i+3] for i in range(0, 9, 3)] if len(flat_list)==9 else None
        
        population = []; attempts = 0
        while len(population) < pop_size and attempts < pop_size * 5:
             state = generate_random_solvable_state(self.goal_tuple);
             if state and is_valid_state(state): population.append(state)
             attempts += 1
        if len(population) < pop_size // 2: 
            print(f"GA: Error generating pop (got {len(population)}). Aborting."); 
            return None # Trả về None nếu không tạo đủ quần thể
            
        best_solution_overall = None
        best_heuristic_overall = float('inf')

        for generation in range(MAX_GA_GENERATIONS):
            pop_fit = []
            for state in population:
                h = self.heuristic(state)
                if h == float('inf'): continue # Bỏ qua trạng thái không hợp lệ/heuristic vô cực
                pop_fit.append({'state': state, 'heuristic': h})
                # Không cần cập nhật best_solution_overall ở đây nữa, sẽ làm sau khi sort pop_fit

            if not pop_fit: 
                print(f"GA: Error - invalid pop gen {generation}, no fittable individuals."); 
                # Nếu quần thể không còn cá thể nào phù hợp, trả về những gì đã có
                if visualization_frames: return visualization_frames
                return [best_solution_overall] if best_solution_overall else None


            pop_fit.sort(key=lambda x: x['heuristic'])
            
            # Lấy cá thể tốt nhất của thế hệ hiện tại để απεικόνιση
            current_gen_best_state = pop_fit[0]['state']
            current_gen_best_h = pop_fit[0]['heuristic']
            visualization_frames.append([r[:] for r in current_gen_best_state]) # <--- THÊM DÒNG NÀY

            if current_gen_best_h < best_heuristic_overall:
                best_heuristic_overall = current_gen_best_h
                best_solution_overall = [r[:] for r in current_gen_best_state] # Cập nhật best_solution_overall
                if best_heuristic_overall == 0:
                    print(f"GA: Solved! Gen {generation}!")
                    # Đảm bảo trạng thái giải được là frame cuối cùng
                    if not visualization_frames or state_to_tuple(visualization_frames[-1]) != state_to_tuple(best_solution_overall):
                        visualization_frames.append([r[:] for r in best_solution_overall])
                    return visualization_frames # <--- THAY ĐỔI: TRẢ VỀ DANH SÁCH FRAME

            next_population = [item['state'] for item in pop_fit[:min(elite_s, len(pop_fit))]]
            
            def tournament_selection(pf, k_val):
                if not pf: return None; k_val = min(k_val, len(pf));
                contenders = random.sample(pf, k_val)
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
                    for index_val in cycle: 
                        if 0 <= index_val < size: child1_flat[index_val] = source1[index_val]; child2_flat[index_val] = source2[index_val]
                child1 = flat_to_state(child1_flat); child2 = flat_to_state(child2_flat);
                if not child1 or not is_valid_state(child1): child1 = p1_state
                if not child2 or not is_valid_state(child2): child2 = p2_state
                return child1, child2

            def mutate(state_to_mutate): 
                mutated_state = [r[:] for r in state_to_mutate]; flat_list = state_to_flat(mutated_state)
                idx1, idx2 = random.sample(range(len(flat_list)), 2)
                flat_list[idx1], flat_list[idx2] = flat_list[idx2], flat_list[idx1];
                new_state = flat_to_state(flat_list)
                return new_state if new_state and is_valid_state(new_state) else state_to_mutate

            while len(next_population) < pop_size:
                parent1 = tournament_selection(pop_fit, tourn_k); parent2 = tournament_selection(pop_fit, tourn_k)
                if not parent1 or not parent2: 
                    # Nếu không chọn được cha mẹ (ví dụ pop_fit quá nhỏ), thêm cá thể ngẫu nhiên hợp lệ
                    if len(population) > 0 : next_population.append(random.choice(population))
                    else: # trường hợp hiếm gặp
                         random_s = generate_random_solvable_state(self.goal_tuple)
                         if random_s: next_population.append(random_s)
                    if len(next_population) >= pop_size: break
                    continue

                child1, child2 = cycle_crossover(parent1, parent2)
                if random.random() < mut_rate: child1 = mutate(child1)
                if random.random() < mut_rate: child2 = mutate(child2)
                if child1 and is_valid_state(child1) and len(next_population) < pop_size: next_population.append(child1)
                if child2 and is_valid_state(child2) and len(next_population) < pop_size: next_population.append(child2)
            
            population = next_population
            if not population: # Nếu quần thể trống sau khi tạo thế hệ mới
                print(f"GA: Population became empty at gen {generation}. Stopping.")
                break

            if generation % 10 == 0 or generation == MAX_GA_GENERATIONS - 1:
                 print(f"GA Gen {generation}: Best H so far={best_heuristic_overall}, Current Gen Best H={current_gen_best_h}, Pop Size={len(population)}")
        
        print(f"GA: Max generations ({MAX_GA_GENERATIONS}) reached. Best H: {best_heuristic_overall}")
        
        # Đảm bảo best_solution_overall (nếu có) là frame cuối cùng
        if best_solution_overall:
            if not visualization_frames or state_to_tuple(visualization_frames[-1]) != state_to_tuple(best_solution_overall):
                visualization_frames.append([r[:] for r in best_solution_overall])
            return visualization_frames # <--- THAY ĐỔI: TRẢ VỀ DANH SÁCH FRAME
        elif visualization_frames: # Nếu không có best_solution_overall nhưng có frames
            return visualization_frames
        else: # Trường hợp không có frame nào và không có best_solution_overall
            return None # Hoặc một list rỗng tùy theo cách xử lý ở _process_solver_result

    def belief_state_search(self, initial_belief_states_lol):
        print("Belief State Search: Starting...")
        current_belief_states_tuples = set()
        for state_lol_item in initial_belief_states_lol:
            if is_valid_state(state_lol_item):
                 current_belief_states_tuples.add(state_to_tuple(state_lol_item))
            else: print(f"BSS Warning: Initial belief state invalid and skipped: {state_lol_item}")
        if not current_belief_states_tuples:
            print("Belief State Search: Failed (No valid initial belief states)."); return []
        history_of_belief_states = [list(current_belief_states_tuples)]
        MAX_BSS_STEPS = 1000; visited_belief_states_hashes = {frozenset(current_belief_states_tuples)}
        for step_count in range(MAX_BSS_STEPS):
            print(f"BSS Step {step_count + 1}: Current belief states count = {len(current_belief_states_tuples)}")
            if len(current_belief_states_tuples) == 1:
                single_state_tuple = list(current_belief_states_tuples)[0]
                if single_state_tuple == self.goal_tuple:
                    print(f"Belief State Search: Solved! Converged to goal in {step_count + 1} steps.")
                    return history_of_belief_states
            if not current_belief_states_tuples:
                print("Belief State Search: Failed (Belief state became empty)."); return history_of_belief_states
            possible_actions_this_iteration = set()
            for state_tuple_in_belief in current_belief_states_tuples:
                r_blank, c_blank = -1, -1
                for r_idx, row_tuple in enumerate(state_tuple_in_belief):
                    try: c_blank = row_tuple.index(0); r_blank = r_idx; break
                    except ValueError: continue
                if r_blank == -1: print(f"BSS Warning: Blank not found in state_tuple {state_tuple_in_belief}"); continue
                if r_blank > 0: possible_actions_this_iteration.add("UP")
                if r_blank < self.rows - 1: possible_actions_this_iteration.add("DOWN")
                if c_blank > 0: possible_actions_this_iteration.add("LEFT")
                if c_blank < self.cols - 1: possible_actions_this_iteration.add("RIGHT")
            if not possible_actions_this_iteration:
                print("Belief State Search: Failed (No possible actions)."); return history_of_belief_states
            next_belief_states_tuples = set()
            for action_str in possible_actions_this_iteration:
                for current_s_tuple in current_belief_states_tuples:
                    current_s_lol = [list(r) for r in current_s_tuple]
                    r_blank_curr, c_blank_curr = -1, -1
                    for r_idx, row_val in enumerate(current_s_lol):
                        try: c_blank_curr = row_val.index(0); r_blank_curr = r_idx; break
                        except ValueError: continue
                    if r_blank_curr == -1 : continue # Should not happen if blank check above passed
                    new_state_lol = [row[:] for row in current_s_lol]; moved_successfully = False
                    if action_str == "UP" and r_blank_curr > 0:
                        new_state_lol[r_blank_curr][c_blank_curr], new_state_lol[r_blank_curr-1][c_blank_curr] = new_state_lol[r_blank_curr-1][c_blank_curr], new_state_lol[r_blank_curr][c_blank_curr]; moved_successfully = True
                    elif action_str == "DOWN" and r_blank_curr < self.rows - 1:
                        new_state_lol[r_blank_curr][c_blank_curr], new_state_lol[r_blank_curr+1][c_blank_curr] = new_state_lol[r_blank_curr+1][c_blank_curr], new_state_lol[r_blank_curr][c_blank_curr]; moved_successfully = True
                    elif action_str == "LEFT" and c_blank_curr > 0:
                        new_state_lol[r_blank_curr][c_blank_curr], new_state_lol[r_blank_curr][c_blank_curr-1] = new_state_lol[r_blank_curr][c_blank_curr-1], new_state_lol[r_blank_curr][c_blank_curr]; moved_successfully = True
                    elif action_str == "RIGHT" and c_blank_curr < self.cols - 1:
                        new_state_lol[r_blank_curr][c_blank_curr], new_state_lol[r_blank_curr][c_blank_curr+1] = new_state_lol[r_blank_curr][c_blank_curr+1], new_state_lol[r_blank_curr][c_blank_curr]; moved_successfully = True
                    if moved_successfully: next_belief_states_tuples.add(state_to_tuple(new_state_lol))
            current_belief_states_tuples = next_belief_states_tuples
            current_belief_hash = frozenset(current_belief_states_tuples)
            if current_belief_hash in visited_belief_states_hashes:
                print("Belief State Search: Failed (Repeated entire belief state).")
                history_of_belief_states.append(list(current_belief_states_tuples)); return history_of_belief_states
            visited_belief_states_hashes.add(current_belief_hash)
            history_of_belief_states.append(list(current_belief_states_tuples))
        print(f"Belief State Search: Failed (Max steps {MAX_BSS_STEPS} reached).")
        return history_of_belief_states
    
    def get_misplaced_tiles_heuristic(self, state_board_list_of_lists):
        """
        Tính heuristic số ô sai vị trí cho một trạng thái board (dạng list of lists).
        """
        misplaced = 0
        # state_board_list_of_lists là list of lists [[],[],[]]
        current_tuple = state_to_tuple(state_board_list_of_lists)
        goal_tuple = self.goal_tuple # self.goal_tuple đã được định nghĩa là tuple of tuples

        for r in range(self.rows):
            for c in range(self.cols):
                if current_tuple[r][c] != 0 and current_tuple[r][c] != goal_tuple[r][c]:
                    misplaced += 1
        return misplaced

    def q_learning(self, callback=None, heuristic='manhattan',
                   alpha=0.1, gamma=0.9, epsilon=0.1, episodes=1000, max_steps=100):
        """
        Triển khai thuật toán Q-learning để giải bài toán 8-puzzle.
        Sử dụng self.start làm trạng thái bắt đầu.
        """
        import random
        import numpy as np
        from collections import defaultdict
        # 'time' đã được import ở đầu file chính

        # Ghi chú: Hàm này phụ thuộc vào sự tồn tại của lớp PuzzleState với
        # các thuộc tính .board, .move và phương thức .get_children().

        start_time = time.time()

        # Chọn hàm heuristic
        if heuristic == 'manhattan':
            # self.heuristic là manhattan distance và nhận list of lists
            h_func = lambda state_obj: self.heuristic(state_obj.board)
        else: # 'misplaced'
            h_func = lambda state_obj: self.get_misplaced_tiles_heuristic(state_obj.board)

        # Khởi tạo Q-table
        Q = defaultdict(float)

        # Theo dõi thống kê
        nodes_expanded = 0
        total_rewards = []
        best_solution_path_states = None # Sẽ lưu trữ list các PuzzleState objects
        best_solution_length = float('inf')

        # Hàm tạo key cho Q-table từ PuzzleState object
        def state_to_key(state_obj): # state_obj là PuzzleState
            return str(state_obj.board) # Sử dụng .board của PuzzleState

        # Lấy các hành động hợp lệ cho một PuzzleState object
        def get_valid_actions(state_obj): # state_obj là PuzzleState
            # .get_children() trả về list các PuzzleState con
            # mỗi con có .move là hành động (ví dụ: "UP", "DOWN")
            return [child.move for child in state_obj.get_children()]

        # Chọn hành động sử dụng epsilon-greedy
        def select_action(state_obj, valid_actions_list): # state_obj là PuzzleState
            current_state_key = state_to_key(state_obj)
            if random.random() < epsilon:
                return random.choice(valid_actions_list)
            else:
                q_values = [Q[(current_state_key, act)] for act in valid_actions_list]
                max_q = -float('inf')
                # Tìm max_q thủ công để tránh lỗi nếu q_values rỗng (dù get_valid_actions nên đảm bảo không rỗng)
                if q_values:
                    max_q = max(q_values)

                # Xử lý nhiều hành động có cùng giá trị Q tối đa
                best_actions = [act for act, q_val in zip(valid_actions_list, q_values) if q_val == max_q]
                if not best_actions: # Nếu không có hành động nào (rất hiếm, có thể do lỗi logic)
                    return random.choice(valid_actions_list) if valid_actions_list else None
                return random.choice(best_actions)

        # Thực hiện hành động và lấy PuzzleState tiếp theo
        def take_action(state_obj, action_to_take): # state_obj là PuzzleState
            children_states = state_obj.get_children()
            for child_state in children_states:
                if child_state.move == action_to_take:
                    return child_state
            return None # Không nên xảy ra nếu action_to_take hợp lệ

        # Vòng lặp huấn luyện
        initial_epsilon = epsilon # Lưu epsilon ban đầu để có thể decay
        for episode in range(episodes):
            # Bắt đầu từ self.start (list of lists)
            # Tạo PuzzleState object từ self.start
            current_state_obj = PuzzleState([row[:] for row in self.start], puzzle_instance=self)
            episode_rewards = 0
            current_path_episode = [current_state_obj] # Path cho episode hiện tại

            for step in range(max_steps):
                nodes_expanded += 1

                # self.is_goal nhận board (list of lists)
                if self.is_goal(current_state_obj.board):
                    episode_rewards += 100 # Phần thưởng lớn khi đạt đích
                    if len(current_path_episode) < best_solution_length:
                        best_solution_path_states = current_path_episode[:]
                        best_solution_length = len(current_path_episode)
                    break # Kết thúc episode

                valid_actions = get_valid_actions(current_state_obj)
                if not valid_actions:
                    break # Không có nước đi hợp lệ

                action = select_action(current_state_obj, valid_actions)
                if action is None: # Không có hành động nào được chọn (hiếm)
                    break

                next_state_obj = take_action(current_state_obj, action)
                if next_state_obj is None: # Không thể thực hiện hành động (lỗi)
                    print(f"Warning: take_action returned None for state {current_state_obj.board} and action {action}")
                    break


                # Tính phần thưởng
                reward = -1 # Chi phí mỗi bước
                current_h_val = h_func(current_state_obj)
                next_h_val = h_func(next_state_obj)
                if next_h_val < current_h_val:
                    reward += 0.5 # Phần thưởng nhỏ nếu tiến gần hơn đến đích

                # Cập nhật Q-value
                current_q_key = state_to_key(current_state_obj)
                current_q_val = Q[(current_q_key, action)]

                if self.is_goal(next_state_obj.board):
                    max_next_q = 0
                    reward = 100 # Phần thưởng lớn khi đạt đích trong bước này
                else:
                    next_valid_actions = get_valid_actions(next_state_obj)
                    if not next_valid_actions:
                        max_next_q = 0 # Trạng thái cuối không có hành động
                    else:
                        # default=0 nếu next_valid_actions rỗng (mặc dù không nên)
                        max_next_q = max([Q[(state_to_key(next_state_obj), a)] for a in next_valid_actions], default=0)

                new_q_val = current_q_val + alpha * (reward + gamma * max_next_q - current_q_val)
                Q[(current_q_key, action)] = new_q_val

                current_state_obj = next_state_obj
                current_path_episode.append(current_state_obj)
                episode_rewards += reward

                if callback and episode == episodes - 1: # Chỉ hiển thị episode cuối
                    # callback mong đợi state (list of lists), nodes_expanded, 0, time
                    callback(current_state_obj.board, nodes_expanded, 0, time.time() - start_time)

            total_rewards.append(episode_rewards)
            
            # Decay epsilon, ví dụ:
            epsilon = max(0.01, initial_epsilon * (1 - (episode / episodes))) # Tuyến tính
            # Hoặc epsilon = max(0.01, epsilon * 0.99) # Hàm gốc

        # Sau khi huấn luyện, tạo đường đi giải pháp bằng chính sách đã học
        # (Không khám phá nữa, epsilon = 0)
        if not best_solution_path_states: # Nếu chưa tìm thấy đường đi trong quá trình huấn luyện
            # Tạo PuzzleState object từ self.start
            current_state_obj_eval = PuzzleState([row[:] for row in self.start], puzzle_instance=self)
            solution_path_eval = [current_state_obj_eval]

            for _ in range(max_steps * 2): # Cho phép nhiều bước hơn để tìm giải pháp
                if self.is_goal(current_state_obj_eval.board):
                    break
                
                valid_actions_eval = get_valid_actions(current_state_obj_eval)
                if not valid_actions_eval:
                    break

                # Chọn hành động tốt nhất dựa trên Q-values (không khám phá)
                current_eval_key = state_to_key(current_state_obj_eval)
                q_values_eval = {act: Q[(current_eval_key, act)] for act in valid_actions_eval}
                
                if not q_values_eval: # Không có Q-value (lạ)
                    break
                best_action_eval = max(q_values_eval, key=q_values_eval.get)

                next_state_obj_eval = take_action(current_state_obj_eval, best_action_eval)
                if next_state_obj_eval is None: break

                current_state_obj_eval = next_state_obj_eval
                solution_path_eval.append(current_state_obj_eval)
            
            if solution_path_eval and self.is_goal(solution_path_eval[-1].board):
                best_solution_path_states = solution_path_eval

        end_time = time.time()

        # Trả về kết quả
        # best_solution_path_states là list các PuzzleState objects
        # Chuyển đổi thành list các board (list of lists) nếu cần cho phần còn lại của hệ thống
        # Tuy nhiên, hàm gốc trả về dict, nên ta cũng làm vậy. Path sẽ là list PuzzleState objects.
        if best_solution_path_states and self.is_goal(best_solution_path_states[-1].board):
            return {
                "path": best_solution_path_states, # List các PuzzleState objects
                "nodes_expanded": nodes_expanded,
                "max_queue_size": 1,  # Not applicable for Q-learning
                "time": end_time - start_time,
                "episodes": episodes,
                "final_epsilon": epsilon,
                "avg_reward_per_episode": np.mean(total_rewards) if total_rewards else 0
            }
        else:
            return {
                "path": None,
                "nodes_expanded": nodes_expanded,
                "max_queue_size": 1,
                "time": end_time - start_time,
                "episodes": episodes,
                "final_epsilon": epsilon,
                "avg_reward_per_episode": np.mean(total_rewards) if total_rewards else 0
            }

# --- END Lớp Puzzle ---

# --- Pygame GUI Application ---
class PygamePuzzleApp:
    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("8-Puzzle AI Solver (Pygame)")
        self.clock = pygame.time.Clock()
        self.running = True
        self.buttons = {}
        self.sliders = {}
        self.grid_rect = pygame.Rect(GRID_X, GRID_Y, GRID_SIZE, GRID_SIZE)
        self.results_rect = pygame.Rect(RESULTS_X, RESULTS_Y, RESULTS_WIDTH, RESULTS_HEIGHT)
        self.message_rect = pygame.Rect(CONTROLS_X, CONTROLS_Y, CONTROLS_WIDTH, 50)
        self.controls_rect = pygame.Rect(CONTROLS_X, CONTROLS_Y + 60, CONTROLS_WIDTH, HEIGHT - CONTROLS_Y - 70 - BOTTOM_MARGIN)
        self.bottom_panel_rect = pygame.Rect(BOTTOM_PANEL_X, BOTTOM_PANEL_Y, BOTTOM_PANEL_WIDTH, BOTTOM_PANEL_HEIGHT)
        self.app_state = "INPUT" # INPUT, READY, FAILED, PLACING_RANDOM, RUNNING, VISUALIZING, VISUALIZING_BELIEF, BENCHMARKING, SOLVED
        self.current_grid_state = [[0] * 3 for _ in range(3)]
        self.num_to_place = 1
        self.user_start_state = None
        self.selected_algorithm_name = None
        self.selected_algorithm_text = None
        self.solution_path = None
        self.vis_step_index = 0
        self.puzzle_instance = None
        self.message = ""
        self.message_color = MSG_DEFAULT_COLOR
        self.final_state_after_vis = None
        self.placing_target_state = None
        self.placing_current_number = 0
        self.result_queue = queue.Queue()
        self.results_lines = []
        self.results_scroll_offset = 0
        self.results_total_height = 0
        self.results_line_height = RESULTS_FONT.get_linesize()
        self.vis_delay = 175
        self.last_vis_time = 0
        self.placement_delay = 200
        self.last_placement_time = 0
        self.hovered_button_name = None
        self.active_slider_name = None

        self.buttons_config = {
            'BFS': {'func_name': 'bfs', 'type': 'path'}, 'UCS': {'func_name': 'ucs', 'type': 'path'},
            'DFS': {'func_name': 'dfs', 'type': 'path'}, 'IDDFS': {'func_name': 'iddfs', 'type': 'path'},
            'Greedy': {'func_name': 'greedy', 'type': 'path'}, 'A*': {'func_name': 'a_star', 'type': 'path'},
            'IDA*': {'func_name': 'ida_star', 'type': 'path'},
            'Belief': {'func_name': 'belief_state_search', 'type': 'belief_explore'},
            'Backtrack': {'func_name': 'backtracking', 'type': 'generate_explore'},
            'CSP Solve': {'func_name': 'csp_solve', 'type': 'generate_explore'},
            'and_or': {'func_name': 'and_or_search', 'type': 'path_if_found'},
            'Spl HC': {'func_name': 'simple_hill_climbing', 'type': 'local'},
            'Stee HC': {'func_name': 'steepest_ascent_hill_climbing', 'type': 'local'},
            'Stoch HC': {'func_name': 'stochastic_hill_climbing', 'type': 'local'},
            'SA': {'func_name': 'simulated_annealing', 'type': 'local'},
            'Beam': {'func_name': 'beam_search', 'type': 'path'},
            'Genetic': {'func_name': 'genetic_algorithm_solve', 'type': 'local'},
            'Q-Learning': {'func_name': 'q_learning', 'type': 'path'},
            'Benchmark': {'func_name': 'benchmark', 'type': 'action'},
            'Reset': {'func_name': 'reset', 'type': 'action'},
        }
        self.pathfinding_algos_for_benchmark = [
            'bfs', 'ucs', 'dfs', 'iddfs', 'greedy', 'a_star', 'ida_star', 'beam_search'
        ]
        self.algo_name_map = {data['func_name']: text for text, data in self.buttons_config.items()}
        self._create_ui_elements()
        self.reset_app()

    def _create_ui_elements(self):
        # --- Kích thước và khoảng cách nút ---
        # Thử nghiệm với 3 nút trên một hàng
        num_buttons_per_row = 3
        button_width = (self.controls_rect.width - PANEL_PADDING * (num_buttons_per_row -1) - PANEL_PADDING) // num_buttons_per_row
        button_height = 35
        button_spacing_x = PANEL_PADDING
        button_spacing_y = 5 # Khoảng cách dọc giữa các hàng nút

        # --- Nút Reset và Benchmark ---
        # Đặt các nút này ở dưới cùng của controls_rect trước
        action_button_height = 35 # Có thể khác với button_height của algo
        reset_rect_height = action_button_height
        benchmark_rect_height = action_button_height

        # Vị trí Y cho nút Reset (ở dưới cùng)
        reset_y = self.controls_rect.bottom - reset_rect_height - button_spacing_y
        # Vị trí Y cho nút Benchmark (phía trên Reset)
        benchmark_y = reset_y - benchmark_rect_height - button_spacing_y

        reset_cfg = self.buttons_config['Reset']
        reset_rect = pygame.Rect(
            self.controls_rect.x + PANEL_PADDING // 2,
            reset_y,
            self.controls_rect.width - PANEL_PADDING, # Chiếm toàn bộ chiều rộng trừ padding
            reset_rect_height
        )
        self.buttons['Reset'] = {
            'rect': reset_rect, 'text': 'Reset',
            'func_name': reset_cfg['func_name'], 'type': reset_cfg['type'],
            'enabled': True
        }

        bench_cfg = self.buttons_config['Benchmark']
        bench_rect = pygame.Rect(
            self.controls_rect.x + PANEL_PADDING // 2,
            benchmark_y,
            self.controls_rect.width - PANEL_PADDING, # Chiếm toàn bộ chiều rộng
            benchmark_rect_height
        )
        self.buttons['Benchmark'] = {
            'rect': bench_rect, 'text': 'Benchmark',
            'func_name': bench_cfg['func_name'], 'type': bench_cfg['type'],
            'enabled': False # Sẽ được cập nhật bởi update_button_states
        }

        # --- Nút Thuật Toán ---
        # Khu vực còn lại cho các nút thuật toán là từ đỉnh controls_rect đến trên nút Benchmark
        algo_buttons_area_top = self.controls_rect.y + button_spacing_y
        algo_buttons_area_bottom = benchmark_y - button_spacing_y # Khoảng trống phía trên Benchmark
        
        algo_buttons_config = []
        for text, config in self.buttons_config.items():
            if config['type'] != 'action': # Chỉ lấy các nút thuật toán
                algo_buttons_config.append((text, config))
        
        # Sắp xếp các nút thuật toán theo tên để thứ tự ổn định (tùy chọn)
        # algo_buttons_config.sort(key=lambda item: item[0])

        current_x = self.controls_rect.x + PANEL_PADDING // 2
        current_y = algo_buttons_area_top
        col_count = 0

        for text, config in algo_buttons_config:
            if current_y + button_height > algo_buttons_area_bottom:
                # Không đủ không gian cho hàng nút mới, có thể dừng ở đây hoặc xử lý khác
                print(f"Warning: Not enough space for all algorithm buttons. Button '{text}' might be hidden.")
                break # Dừng thêm nút nếu không đủ chỗ

            rect = pygame.Rect(current_x, current_y, button_width, button_height)
            self.buttons[text] = {
                'rect': rect, 'text': text,
                'func_name': config['func_name'], 'type': config['type'],
                'enabled': False # Sẽ được cập nhật bởi update_button_states
            }

            col_count += 1
            if col_count >= num_buttons_per_row:
                col_count = 0
                current_x = self.controls_rect.x + PANEL_PADDING // 2
                current_y += button_height + button_spacing_y
            else:
                current_x += button_width + button_spacing_x
        
        # --- Sliders (giữ nguyên logic slider của bạn) ---
        slider_section_width = BOTTOM_PANEL_WIDTH // 2 - PANEL_PADDING * 1.5 # Đảm bảo BOTTOM_PANEL_WIDTH được định nghĩa đúng
        slider_width = slider_section_width - 10
        slider_height = 15
        knob_width = 10
        knob_height = 20
        slider_start_x = self.bottom_panel_rect.x + PANEL_PADDING
        slider_y_base = self.bottom_panel_rect.y + 10 # Y cơ sở cho slider đầu tiên

        vis_rect = pygame.Rect(slider_start_x, slider_y_base, slider_width, slider_height)
        vis_knob_rect = pygame.Rect(0, 0, knob_width, knob_height) # Sẽ được cập nhật vị trí sau
        self.sliders['vis_speed'] = {
            'rect': vis_rect, 'knob_rect': vis_knob_rect,
            'min': 10, 'max': 1000, 'value': self.vis_delay, # Sử dụng self.vis_delay đã có
            'label': "Vis Speed(ms)", 'dragging': False
        }

        depth_slider_y = slider_y_base + knob_height + 15 # Khoảng cách giữa các slider
        depth_rect = pygame.Rect(slider_start_x, depth_slider_y, slider_width, slider_height)
        depth_knob_rect = pygame.Rect(0, 0, knob_width, knob_height)
        self.sliders['max_depth'] = {
            'rect': depth_rect, 'knob_rect': depth_knob_rect,
            'min': 1, 'max': 30, 'value': DEFAULT_MAX_DEPTH, # Sử dụng DEFAULT_MAX_DEPTH
            'label': "Max Depth", 'dragging': False
        }
        self._update_knob_position('vis_speed')
        self._update_knob_position('max_depth')

    def draw_grid(self):
        pygame.draw.rect(self.screen, DARK_GREY, self.grid_rect, border_radius=5)
        for r in range(3):
            for c in range(3):
                num = self.current_grid_state[r][c]
                tile_rect = pygame.Rect(GRID_X + c*TILE_SIZE + 2, GRID_Y + r*TILE_SIZE + 2, TILE_SIZE-4, TILE_SIZE-4)
                if num == 0: pygame.draw.rect(self.screen, TILE_EMPTY_BG, tile_rect, border_radius=3)
                elif num == -1: pygame.draw.rect(self.screen, DARK_RED, tile_rect, border_radius=3)
                else:
                    color_index = (num - 1) % len(TILE_COLORS)
                    pygame.draw.rect(self.screen, TILE_COLORS[color_index], tile_rect, border_radius=3)
                    text_surf = TILE_FONT.render(str(num), True, WHITE)
                    text_rect = text_surf.get_rect(center=tile_rect.center)
                    self.screen.blit(text_surf, text_rect)
                pygame.draw.rect(self.screen, TILE_BORDER, tile_rect, 1, border_radius=3)

    def draw_buttons(self):
        mouse_pos = pygame.mouse.get_pos(); self.hovered_button_name = None
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
            lines = textwrap.wrap(self.message, width=35); y_offset = self.message_rect.y + 5
            for i, line in enumerate(lines):
                 if y_offset + MSG_FONT.get_linesize() > self.message_rect.bottom : break
                 msg_surf = MSG_FONT.render(line, True, self.message_color)
                 msg_rect = msg_surf.get_rect(centerx=self.message_rect.centerx, top=y_offset)
                 if msg_rect.width > self.message_rect.width - 10: msg_rect.left = self.message_rect.left + 5
                 self.screen.blit(msg_surf, msg_rect); y_offset += MSG_FONT.get_linesize()

    def draw_results(self):
        pygame.draw.rect(self.screen, WHITE, self.results_rect)
        pygame.draw.rect(self.screen, DARK_GREY, self.results_rect, 1)
        y = self.results_rect.y + 5 - self.results_scroll_offset
        max_y = self.results_rect.bottom - 5
        original_clip = self.screen.get_clip()
        self.screen.set_clip(self.results_rect.inflate(-4, -4))
        self.results_total_height = 0
        for i, line in enumerate(self.results_lines):
             line_height = self.results_line_height; self.results_total_height += line_height
             if y + line_height > self.results_rect.top and y < max_y:
                 try:
                      res_surf = RESULTS_FONT.render(line, True, TEXT_COLOR)
                      res_rect = res_surf.get_rect(topleft=(self.results_rect.x + 5, y))
                      self.screen.blit(res_surf, res_rect)
                 except Exception as e: print(f"Error rendering line '{line}': {e}")
             y += line_height
             if y >= max_y + self.results_scroll_offset : break
        self.screen.set_clip(original_clip)
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
        if slider_name == 'max_depth': new_value = int(round(new_value))
        elif slider_name == 'vis_speed': new_value = int(round(new_value))
        if abs(new_value - old_value) > 1e-6:
             data['value'] = new_value; self._update_knob_position(slider_name)
             if slider_name == 'vis_speed': self.vis_delay = int(data['value'])
             elif slider_name == 'max_depth':
                  current_depth = int(data['value'])
                  if self.puzzle_instance: self.puzzle_instance.iddfs_max_depth = current_depth
                  global DEFAULT_MAX_DEPTH; DEFAULT_MAX_DEPTH = current_depth
             return True
        return False

    def draw_sliders(self):
        for name, data in self.sliders.items():
            track_rect = data['rect']; knob_rect = data['knob_rect']
            pygame.draw.rect(self.screen, SLIDER_TRACK_COLOR, track_rect, border_radius=5)
            pygame.draw.rect(self.screen, SLIDER_KNOB_COLOR, knob_rect, border_radius=3)
            pygame.draw.rect(self.screen, SLIDER_KNOB_BORDER, knob_rect, 1, border_radius=3)
            label_text = f"{data['label']}: {int(data['value'])}" # Ensure int display
            label_surf = SLIDER_FONT.render(label_text, True, TEXT_COLOR)
            label_rect = label_surf.get_rect(midleft=(track_rect.right + 10, track_rect.centery))
            if label_rect.right > WIDTH - PANEL_PADDING: label_rect.midbottom = (track_rect.centerx, track_rect.top - 2)
            self.screen.blit(label_surf, label_rect)

    def draw_notes(self):
        note_y = self.bottom_panel_rect.y + 5
        note_x = self.bottom_panel_rect.x + self.bottom_panel_rect.width // 2 + PANEL_PADDING
        notes = ["'Benchmark' compares pathfinders.", "'Reset' clears board & results.",
                 "'Backtrack' generates & solves.", "Use sliders for Vis Speed & Depth."]
        for note in notes:
             if note_y + NOTE_FONT.get_linesize() > self.bottom_panel_rect.bottom: break
             note_surf = NOTE_FONT.render(note, True, NOTE_COLOR)
             note_rect = note_surf.get_rect(topleft=(note_x, note_y))
             if note_rect.right > WIDTH - PANEL_PADDING: continue
             self.screen.blit(note_surf, note_rect); note_y += NOTE_FONT.get_linesize() + 2

    def set_message(self, text, color=MSG_DEFAULT_COLOR):
        self.message = text; self.message_color = color; print(f"MSG: {text}")

    def set_results(self, lines):
        self.results_lines = lines[:]; self.results_scroll_offset = 0
        self.results_total_height = sum(self.results_line_height for _ in lines)

    def update_grid_display(self, state):
         try:
             new_state_copy = [list(row) for row in state]
             if len(new_state_copy) != 3 or any(len(r) != 3 for r in new_state_copy):
                 self.current_grid_state = [[-1]*3 for _ in range(3)]
             else: self.current_grid_state = new_state_copy
         except (TypeError, IndexError): self.current_grid_state = [[-1]*3 for _ in range(3)]

    def update_button_states(self):
        start_state_is_ready = False
        if self.user_start_state:
            try: start_state_is_ready = is_solvable(self.user_start_state)
            except Exception: pass
        can_interact = self.app_state not in ["PLACING_RANDOM", "RUNNING", "VISUALIZING", "VISUALIZING_BELIEF", "BENCHMARKING"]
        for name, data in self.buttons.items():
            enabled = False; func_name = data['func_name']; algo_type = data['type']
            if can_interact:
                if algo_type in ['path', 'explore_path', 'path_if_found', 'local', 'belief_explore']: # Added belief_explore
                    enabled = start_state_is_ready
                elif algo_type in ['generate_explore', 'state_only', 'action']:
                    enabled = True
                    if func_name == 'benchmark': enabled = start_state_is_ready
            if func_name == 'reset': enabled = can_interact # Reset always enabled if not busy
            data['enabled'] = enabled

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
                    else:
                        for name, data in self.sliders.items():
                             if data['knob_rect'].collidepoint(mouse_pos) or data['rect'].collidepoint(mouse_pos):
                                 self.active_slider_name = name; data['dragging'] = True
                                 self._update_slider_value_from_pos(name, mouse_pos[0]); break
            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    if self.active_slider_name:
                        self.sliders[self.active_slider_name]['dragging'] = False; self.active_slider_name = None
            if event.type == pygame.MOUSEMOTION:
                 if self.active_slider_name and self.sliders[self.active_slider_name]['dragging']: self._update_slider_value_from_pos(self.active_slider_name, mouse_pos[0])
                 else:
                      self.hovered_button_name = None
                      for name, data in self.buttons.items():
                           if data['rect'].collidepoint(mouse_pos) and data['enabled']: self.hovered_button_name = name; break

    def _handle_grid_click(self, mouse_pos):
        if self.app_state != "INPUT": return
        col = (mouse_pos[0] - GRID_X) // TILE_SIZE; row = (mouse_pos[1] - GRID_Y) // TILE_SIZE
        row = max(0, min(2, row)); col = max(0, min(2, col))
        if self.current_grid_state[row][col] == 0:
            if self.num_to_place <= 8:
                self.current_grid_state[row][col] = self.num_to_place; placed_number = self.num_to_place
                self.num_to_place += 1; self.update_grid_display(self.current_grid_state)
                if self.num_to_place == 9:
                    self.user_start_state = [r[:] for r in self.current_grid_state]
                    if is_valid_state(self.user_start_state):
                        if is_solvable(self.user_start_state):
                            self.set_message("Board ready! Select algorithm.", MSG_SUCCESS_COLOR); self.app_state = "READY"
                        else: self.set_message("Board is unsolvable! Reset.", MSG_WARN_COLOR); self.app_state = "FAILED"
                    else: self.set_message("Error: Final input state is invalid!", MSG_ERROR_COLOR); self.app_state = "FAILED"
                    self.update_button_states()
                else: self.set_message(f"Placed {placed_number}. Click empty cell for {self.num_to_place}.", MSG_DEFAULT_COLOR)
            else: self.set_message("Board full. Select algorithm or Reset.", MSG_INFO_COLOR)
        else:
            occupied_by = self.current_grid_state[row][col]
            if self.num_to_place <= 8: self.set_message(f"Cell occupied by {occupied_by}! Click for {self.num_to_place}.", MSG_WARN_COLOR)
            else: self.set_message(f"Board full (cell has {occupied_by}). Select or Reset.", MSG_INFO_COLOR)

    def _handle_button_click(self, func_name, text, algo_type):
        print(f"Button Clicked: {text} (func: {func_name}, type: {algo_type})")
        if func_name == 'reset': self.reset_app(); return
        if func_name == 'benchmark':
            if self.user_start_state and is_solvable(self.user_start_state): self.run_benchmark_threaded()
            else: self.set_message("Board not ready/unsolvable for Benchmark!", MSG_WARN_COLOR)
            return
        if algo_type == 'generate_explore': self.start_random_placement(); return
        start_s = None; puzzle_params = {}
        if algo_type in ['path', 'local', 'path_if_found', 'explore_path', 'belief_explore']: # Added belief_explore
            if self.user_start_state and is_solvable(self.user_start_state): start_s = [r[:] for r in self.user_start_state]
            else: self.set_message("Error: Board not set or unsolvable!", MSG_ERROR_COLOR); return
        elif algo_type == 'state_only':
            self.set_message(f"{text} generating state...", MSG_INFO_COLOR); start_s = generate_random_solvable_state()
            if not start_s: self.set_message("Error generating random state for GA!", MSG_ERROR_COLOR); return
            self.update_grid_display(start_s); self.draw(); pygame.display.flip(); time.sleep(0.1)
        else: self.set_message(f"Error: Unknown button type '{algo_type}'!", MSG_ERROR_COLOR); return
        if func_name in ['iddfs', 'and_or_search']:
             puzzle_params['iddfs_max_depth'] = int(self.sliders['max_depth']['value'])
        if start_s: self.run_algorithm_threaded(func_name, text, start_s, puzzle_params)
        else: self.set_message("Error: No valid start state.", MSG_ERROR_COLOR)

    def reset_app(self):
        print("\n--- Resetting Application ---")
        while not self.result_queue.empty():
            try: self.result_queue.get_nowait()
            except queue.Empty: break
        self.current_grid_state = [[0] * 3 for _ in range(3)]; self.num_to_place = 1
        self.user_start_state = None; self.selected_algorithm_name = None
        self.selected_algorithm_text = None; self.solution_path = None
        self.vis_step_index = 0; self.puzzle_instance = None
        self.final_state_after_vis = None; self.placing_target_state = None
        self.placing_current_number = 0; self.results_lines = []
        self.results_scroll_offset = 0; self.results_total_height = 0
        self.app_state = "INPUT"
        self.set_message("Click empty grid cell to place number 1.", MSG_DEFAULT_COLOR)
        self.update_grid_display(self.current_grid_state); self.update_button_states()

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
                for r_idx in range(3): # Renamed r to r_idx
                    for c_idx in range(3): # Renamed c to c_idx
                        if self.placing_target_state[r_idx][c_idx] == self.placing_current_number: target_r, target_c = r_idx, c_idx; found_pos = True; break
                    if found_pos: break
                if found_pos:
                    self.current_grid_state[target_r][target_c] = self.placing_current_number
                    self.update_grid_display(self.current_grid_state)
                    self.set_message(f"Placing number {self.placing_current_number}...", MSG_INFO_COLOR)
                    self.placing_current_number += 1
                else:
                    self.set_message("Error generating board!", MSG_ERROR_COLOR); self.app_state = "FAILED"; self.update_button_states(); self.placing_target_state = None
            else:
                try:
                    br, bc = Puzzle(self.placing_target_state).get_blank_position(self.placing_target_state)
                    if self.current_grid_state[br][bc] != 0:
                         for r0, row0 in enumerate(self.current_grid_state):
                              try: c0 = row0.index(0); self.current_grid_state[r0][c0] = self.placing_target_state[r0][c0]; break
                              except ValueError: continue
                         self.current_grid_state[br][bc] = 0
                    self.update_grid_display(self.current_grid_state)
                except Exception: pass
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
                 if hasattr(self.puzzle_instance, param): setattr(self.puzzle_instance, param, value)
        except Exception as e: self.set_message(f"Error creating puzzle: {e}", MSG_ERROR_COLOR); self.app_state = "FAILED"; self.update_button_states(); return
        thread = threading.Thread(target=self._solver_thread_func, args=(self.puzzle_instance, algo_func_name, algo_display_text, self.result_queue), daemon=True)
        thread.start()

    def _solver_thread_func(self, puzzle_obj, algo_func_name, algo_display_text, q):
        result_data = {'status': 'error', 'message': 'Unknown thread error', 'path': None, 'time': 0, 'algo_name': algo_func_name, 'algo_text': algo_display_text}
        try:
            solver_method = getattr(puzzle_obj, algo_func_name, None)
            if solver_method:
                t_start = time.perf_counter(); temp_result = None
                try:
                    if algo_func_name == 'belief_state_search':
                        initial_belief_for_bss = [[r[:] for r in puzzle_obj.start]]
                        temp_result = solver_method(initial_belief_for_bss) 
                    elif algo_func_name == 'q_learning':
                            # q_learning không nhận initial_state vì nó dùng self.start
                        # Nó trả về một dict chứa path và các thông tin khác
                        temp_result_dict = solver_method() # Gọi puzzle_obj.q_learning()
                        t_elapsed = time.perf_counter() - t_start
                        result_data['time'] = t_elapsed

                        if temp_result_dict and isinstance(temp_result_dict, dict):
                            result_data['path'] = temp_result_dict.get('path') # path là list các PuzzleState objects
                            # Lưu các thông tin thống kê khác từ Q-learning
                            result_data['q_learning_stats'] = {
                                "nodes_expanded": temp_result_dict.get("nodes_expanded"),
                                "episodes": temp_result_dict.get("episodes"),
                                "final_epsilon": temp_result_dict.get("final_epsilon"),
                                "avg_reward_per_episode": temp_result_dict.get("avg_reward_per_episode")
                            }
                            if result_data['path']: # Nếu có đường đi
                                final_state_in_path_obj = result_data['path'][-1] # Là một PuzzleState object
                                # puzzle_obj.is_goal nhận board (list of lists)
                                if puzzle_obj.is_goal(final_state_in_path_obj.board):
                                    result_data['status'] = 'success_goal'
                                    result_data['message'] = f"{algo_display_text} solved."
                                else:
                                    result_data['status'] = 'success_nogoal'
                                    result_data['message'] = f"{algo_display_text} finished (goal not reached by final path)."
                            else: # Không tìm thấy đường đi sau khi huấn luyện
                                result_data['status'] = 'failure'
                                result_data['message'] = f"{algo_display_text} training complete, but no solution path found."
                        else:
                            result_data['status'] = 'error_type'
                            result_data['message'] = f"{algo_display_text}: Bad result type from Q-learning: {type(temp_result_dict)}."
                    else: # Các thuật toán khác
                        temp_result = solver_method()
                        t_elapsed = time.perf_counter() - t_start
                        result_data['time'] = t_elapsed
                        result_data['path'] = temp_result
                        # ... (phần logic status hiện tại cho các thuật toán khác) ...
                        if temp_result is not None and isinstance(temp_result, list):
                            if not temp_result:
                                result_data['status'] = 'failure'
                                result_data['message'] = f"{algo_display_text} returned empty result."
                            else:
                                algo_type = self.buttons_config[algo_display_text]['type']
                                is_goal_reached_overall = False
                                if algo_type == 'belief_explore':
                                    # ... (logic hiện tại)
                                    pass
                                else:
                                    try:
                                        final_state_data = temp_result[-1]
                                        # Kiểm tra xem final_state_data có phải là PuzzleState không (từ GA)
                                        if isinstance(final_state_data, PuzzleState):
                                            is_goal_reached_overall = puzzle_obj.is_goal(final_state_data.board)
                                        else: # Giả sử là list of lists
                                            is_goal_reached_overall = puzzle_obj.is_goal(final_state_data)
                                    except Exception as e_check_goal:
                                        print(f"Error checking goal for {algo_display_text}: {e_check_goal}")
                                        pass # Để is_goal_reached_overall là False
                                
                                if is_goal_reached_overall:
                                    result_data['status'] = 'success_goal'
                                    result_data['message'] = f"{algo_display_text} solved."
                                else:
                                    result_data['status'] = 'success_nogoal'
                                    result_data['message'] = f"{algo_display_text} finished (no goal)."
                        elif temp_result is None:
                            result_data['status'] = 'failure'
                            result_data['message'] = f"{algo_display_text} failed/found nothing."
                        else: # Loại kết quả không mong muốn
                            result_data['status'] = 'error_type';
                            result_data['message'] = f"{algo_display_text}: Bad result type {type(temp_result)}."
                except MemoryError as me: result_data['time']=time.perf_counter()-t_start; result_data['status']='error_memory'; result_data['message']=f"{algo_display_text}: Memory Error!"
                except Exception as e: result_data['time']=time.perf_counter()-t_start; result_data['status']='error_runtime'; result_data['message']=f"{algo_display_text}: Runtime Error!"; traceback.print_exc()
            else: result_data['message'] = f"Error: Func '{algo_func_name}' not found!"; result_data['status'] = 'error_missing_func'
        except Exception as e: result_data['message'] = f"Thread setup error: {e}"; result_data['status'] = 'error_setup'; traceback.print_exc()
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

    def _process_solver_result(self, result):
        algo_text = result['algo_text']
        algo_config = self.buttons_config.get(algo_text, {})
        algo_type = algo_config.get('type', 'unknown')
        status = result['status']
        message_from_thread = result['message'] # Đổi tên để tránh nhầm lẫn với self.message
        path_data_from_thread = result['path'] # Đây có thể là list các board, hoặc list các PuzzleState
        t_elapsed = result['time']

        result_lines_display = []
        start_visualization = False
        is_q_learning = (algo_text == "Q-Learning") # Kiểm tra nếu là Q-Learning

        # Giai đoạn 1: Xác định trạng thái ứng dụng và thông điệp chính
        if status.startswith('error'):
            self.set_message(message_from_thread, MSG_ERROR_COLOR)
            self.app_state = "FAILED"
            result_lines_display = [f"Status {algo_text}: Error", message_from_thread, f"(Time: {t_elapsed:.3f}s)"]
        elif status == 'failure':
            res_msg = f"{algo_text}: Failed/Stopped ({t_elapsed:.3f}s)"
            self.set_message(res_msg, MSG_WARN_COLOR)
            self.app_state = "FAILED"
            result_lines_display = [f"Status {algo_text}: Failed", message_from_thread, f"(Time: {t_elapsed:.3f}s)"]
            if algo_type == 'generate_explore' and path_data_from_thread: # Ví dụ: Backtrack không tìm thấy giải pháp
                self.solution_path = path_data_from_thread
                steps_exp = len(self.solution_path)
                self.set_message(f"{algo_text}: Stopped ({steps_exp} states, {t_elapsed:.3f}s)", MSG_WARN_COLOR)
                result_lines_display = [f"{algo_text} Stopped ({t_elapsed:.3f}s):", f"Explored {steps_exp} states."]
                self.app_state = "VISUALIZING"
                self.final_state_after_vis = "FAILED"
                start_visualization = True
        elif status.startswith('success'):
            try:
                if not self.puzzle_instance:
                    raise RuntimeError("Puzzle instance missing!")

                self.solution_path = path_data_from_thread
                is_goal = (status == 'success_goal')
                self.final_state_after_vis = "SOLVED" if is_goal else "FAILED"

                # Tiêu đề chung cho kết quả thành công
                main_status_line = f"{algo_text} {'Solved' if is_goal else 'Finished'} ({t_elapsed:.3f}s):"
                result_lines_display.append(main_status_line)

                if algo_type in ['path', 'path_if_found', 'beam_search'] or is_q_learning: # Q-Learning trả về path
                    if not self.solution_path or not isinstance(self.solution_path, list) or len(self.solution_path) == 0:
                        self.set_message(f"{algo_text}: Success status but no valid path data.", MSG_ERROR_COLOR)
                        self.app_state = "FAILED"
                        result_lines_display.append("Error: No valid path data for visualization.")
                    else:
                        steps = len(self.solution_path) - 1
                        # Xác định final_h (có thể không áp dụng cho mọi path_type nếu goal)
                        final_h_display = ""
                        if not is_goal:
                            # Kiểm tra phần tử cuối cùng của path là PuzzleState hay list of lists
                            last_element = self.solution_path[-1]
                            if isinstance(last_element, PuzzleState):
                                final_h_val = self.puzzle_instance.heuristic(last_element.board)
                            else: # Giả sử là list of lists
                                final_h_val = self.puzzle_instance.heuristic(last_element)
                            final_h_display = f", H={final_h_val}"

                        res_msg = f"{algo_text}: {'Solved!' if is_goal else 'Finished'}{final_h_display} ({steps} steps, {t_elapsed:.3f}s)"
                        self.set_message(res_msg, MSG_SUCCESS_COLOR if is_goal else MSG_WARN_COLOR)
                        result_lines_display.append(f"Steps: {steps}")

                        if is_goal:
                            # Chuẩn bị path cho get_solution_moves
                            path_for_moves_display = []
                            if self.solution_path and isinstance(self.solution_path[0], PuzzleState):
                                path_for_moves_display = [ps.board for ps in self.solution_path]
                            else:
                                path_for_moves_display = self.solution_path

                            moves = get_solution_moves(path_for_moves_display)
                            result_lines_display.append("--- Moves ---")
                            result_lines_display.extend([f"{i+1}. {m}" for i, m in enumerate(moves)])
                        else:
                            result_lines_display.append("Path will be shown in animation (goal not reached).")

                        self.app_state = "VISUALIZING"
                        start_visualization = True

                elif algo_type in ['explore_path', 'generate_explore']:
                    steps_exp = len(self.solution_path) if self.solution_path else 0
                    path_len_actual = len(self.solution_path) - 1 if self.solution_path and is_goal else steps_exp
                    status_desc = f"Solved (Path len {path_len_actual})" if is_goal else f"Explored ({steps_exp} states)"
                    res_msg = f"{algo_text}: {status_desc}! ({t_elapsed:.3f}s)"
                    self.set_message(res_msg, MSG_SUCCESS_COLOR if is_goal else MSG_WARN_COLOR)
                    result_lines_display.append(f"{'Found path length' if is_goal else 'Explored states'}: {path_len_actual if is_goal else steps_exp}")
                    self.app_state = "VISUALIZING"
                    start_visualization = True

                elif algo_type == 'belief_explore':
                    num_belief_steps = len(self.solution_path) if self.solution_path else 0
                    final_belief_snapshot = self.solution_path[-1] if num_belief_steps > 0 else []
                    num_states_in_final_belief = len(final_belief_snapshot)
                    status_desc = "Converged to Goal" if is_goal else f"Finished (Final Belief: {num_states_in_final_belief} states)"
                    res_msg = f"{algo_text}: {status_desc}! ({num_belief_steps} belief steps, {t_elapsed:.3f}s)"
                    self.set_message(res_msg, MSG_SUCCESS_COLOR if is_goal else MSG_WARN_COLOR)
                    result_lines_display.append(f"Belief Steps: {num_belief_steps}")
                    if is_goal:
                        result_lines_display.append("Final Belief: 1 state (Goal)")
                    else:
                        result_lines_display.append(f"Final Belief: {num_states_in_final_belief} states")
                        if 0 < num_states_in_final_belief <= 10:
                            result_lines_display.append("  States in final belief:")
                            for i, st_tuple in enumerate(final_belief_snapshot):
                                result_lines_display.append(f"    {i+1}. {st_tuple}")
                                if i >= 9 : result_lines_display.append("    ..."); break
                        elif num_states_in_final_belief > 10:
                             result_lines_display.append(f"  (Showing first 10 of {num_states_in_final_belief} states)")
                             for i, st_tuple in enumerate(final_belief_snapshot[:10]):
                                result_lines_display.append(f"    {i+1}. {st_tuple}")
                             result_lines_display.append("    ...")
                    self.app_state = "VISUALIZING_BELIEF"
                    start_visualization = True

                elif algo_type == 'local': # Ví dụ: Hill Climbing, SA, GA
                    # GA trả về một list các frame, phần tử cuối là best.
                    # Các local search khác trả về một path các trạng thái.
                    if not self.solution_path or not isinstance(self.solution_path, list) or len(self.solution_path) == 0:
                        self.set_message(f"{algo_text}: Success status but no valid path/frame data.", MSG_ERROR_COLOR)
                        self.app_state = "FAILED"
                        result_lines_display.append("Error: No valid data for visualization.")
                    else:
                        steps_local = len(self.solution_path) - 1
                        final_state_element = self.solution_path[-1]
                        if isinstance(final_state_element, PuzzleState):
                            final_state_board = final_state_element.board
                        else: # Giả sử là list of lists
                            final_state_board = final_state_element

                        final_h = self.puzzle_instance.heuristic(final_state_board)
                        status_desc = "Solved" if is_goal else f"Finished (H={final_h})"
                        res_msg = f"{algo_text}: {status_desc} ({steps_local} steps, {t_elapsed:.3f}s)"
                        self.set_message(res_msg, MSG_SUCCESS_COLOR if is_goal else MSG_WARN_COLOR)
                        result_lines_display.append(f"Status: {status_desc}")
                        result_lines_display.append(f"Steps/Iterations: {steps_local}")
                        self.app_state = "VISUALIZING"
                        start_visualization = True
                
                elif algo_type == 'state_only': # (Không dùng nữa, GA trả về list các frame cho 'local')
                     final_state_ga = self.solution_path[0] # Giả định path_or_state[0] là trạng thái tốt nhất
                     final_h_ga = self.puzzle_instance.heuristic(final_state_ga)
                     status_desc = "Solved (Genetic)" if is_goal else f"Finished (Best H={final_h_ga})"
                     res_msg = f"{algo_text}: {status_desc} ({t_elapsed:.3f}s)"
                     self.set_message(res_msg, MSG_SUCCESS_COLOR if is_goal else MSG_WARN_COLOR)
                     self.update_grid_display(final_state_ga)
                     self.app_state = "SOLVED" if is_goal else "FAILED"
                     result_lines_display = [f"{algo_text} Result ({t_elapsed:.3f}s):", status_desc]

                else:
                    raise ValueError(f"Unhandled type '{algo_type}' for success in _process_solver_result.")

            except Exception as proc_err:
                 traceback.print_exc()
                 self.set_message(f"Error processing {algo_text} result: {proc_err}", MSG_ERROR_COLOR)
                 self.app_state = "FAILED"
                 result_lines_display = [f"Status {algo_text}: Processing Error", str(proc_err)]
        
        # Giai đoạn 2: Thêm Q-Learning stats nếu có
        q_stats = result.get('q_learning_stats')
        if q_stats:
            result_lines_display.append("--- Q-Learning Stats ---")
            result_lines_display.append(f"Nodes Expanded (total steps): {q_stats.get('nodes_expanded', 'N/A')}")
            result_lines_display.append(f"Training Episodes: {q_stats.get('episodes', 'N/A')}")
            if q_stats.get('final_epsilon') is not None:
                 result_lines_display.append(f"Final Epsilon: {q_stats['final_epsilon']:.4f}")
            if q_stats.get('avg_reward_per_episode') is not None:
                 result_lines_display.append(f"Avg Reward/Episode: {q_stats['avg_reward_per_episode']:.2f}")

        self.set_results(result_lines_display)

        if start_visualization:
            self.vis_step_index = 0
            self.last_vis_time = pygame.time.get_ticks()
        
        if self.app_state not in ["VISUALIZING", "VISUALIZING_BELIEF"]:
            self.update_button_states()

    def _update_visualization(self):
        if self.app_state not in ["VISUALIZING", "VISUALIZING_BELIEF"] or not self.solution_path:
            return

        now = pygame.time.get_ticks()
        current_delay = int(self.vis_delay)  # Đảm bảo vis_delay là int

        is_belief_vis = (self.app_state == "VISUALIZING_BELIEF")
        # Lấy algo_type từ config để điều chỉnh tốc độ cho explore/generate
        algo_config_vis = self.buttons_config.get(self.selected_algorithm_text, {})
        algo_type_vis = algo_config_vis.get('type', 'path')


        if is_belief_vis:
            current_delay = max(50, current_delay // 2) # Belief steps có thể nhanh hơn
        elif algo_type_vis in ['explore_path', 'generate_explore']: # Cho các thuật toán duyệt cây/không gian trạng thái
            current_delay = max(10, current_delay // 4) # Nhanh hơn nữa
        elif algo_type_vis == 'local': # Cho các thuật toán local search như HC, SA, GA
             current_delay = max(20, current_delay // 2)


        if now - self.last_vis_time > current_delay:
            self.last_vis_time = now

            if self.vis_step_index < len(self.solution_path):
                current_step_data = self.solution_path[self.vis_step_index]
                display_state_for_grid = None
                num_elements_in_current_step = 0 # Cho belief search

                if is_belief_vis:
                    # current_step_data là list các tuple trạng thái
                    num_elements_in_current_step = len(current_step_data)
                    if current_step_data:
                        representative_state_tuple = current_step_data[0] # Lấy trạng thái đầu tiên để hiển thị
                        display_state_for_grid = [list(r) for r in representative_state_tuple]
                    else: # Belief state rỗng (không nên xảy ra nếu logic đúng)
                        display_state_for_grid = [[0]*3 for _ in range(3)] # Hoặc một trạng thái lỗi
                elif isinstance(current_step_data, PuzzleState): # Đối với Q-Learning hoặc các algo trả về PuzzleState
                    display_state_for_grid = current_step_data.board
                elif isinstance(current_step_data, list) and len(current_step_data) == 3 and isinstance(current_step_data[0], list): # list of lists
                    display_state_for_grid = current_step_data
                else: # Loại dữ liệu không mong muốn trong path
                    self.set_message(f"Visualization Error: Unknown data type in path at step {self.vis_step_index}", MSG_ERROR_COLOR)
                    self.app_state = self.final_state_after_vis or "FAILED"
                    self.update_button_states()
                    self.solution_path = None
                    return

                if display_state_for_grid is not None and is_valid_state(display_state_for_grid):
                    self.update_grid_display(display_state_for_grid)
                else:
                    self.set_message(f"Visualization Error: Invalid state at step {self.vis_step_index}", MSG_ERROR_COLOR)
                    self.app_state = self.final_state_after_vis or "FAILED"
                    self.update_button_states()
                    self.solution_path = None
                    return

                # Cập nhật thông điệp visualization
                steps_total_display = len(self.solution_path)
                current_step_display = self.vis_step_index + 1
                msg_prefix = "Step"

                if is_belief_vis:
                    msg_prefix = "Belief Step"
                    self.set_message(f"{msg_prefix}: {current_step_display}/{steps_total_display}, States: {num_elements_in_current_step}", MSG_INFO_COLOR)
                elif algo_type_vis in ['explore_path', 'generate_explore']:
                    msg_prefix = "Explore Step"
                    self.set_message(f"{msg_prefix}: {current_step_display}/{steps_total_display}", MSG_INFO_COLOR)
                elif algo_type_vis == 'local':
                    msg_prefix = "Local Iteration" if self.selected_algorithm_text == "Genetic" else "Local Step"
                    self.set_message(f"{msg_prefix}: {current_step_display}/{steps_total_display}", MSG_INFO_COLOR)
                else: # pathfinding algorithms (BFS, A*, Q-Learning path, etc.)
                    if self.vis_step_index == 0:
                        self.set_message(f"{self.selected_algorithm_text} - Start State", MSG_INFO_COLOR)
                    else:
                        self.set_message(f"{msg_prefix}: {self.vis_step_index}/{steps_total_display -1}", MSG_INFO_COLOR)
                
                self.vis_step_index += 1
            else:  # Visualization finished
                self.app_state = self.final_state_after_vis or "FAILED"
                
                # Lấy thời gian từ results panel nếu có
                time_str_vis = ""
                try:
                     if self.results_lines:
                         for line in self.results_lines[:3]: # Kiểm tra vài dòng đầu
                             time_part_match = re.search(r"\((\d+\.\d+)s\)", line)
                             if time_part_match:
                                 time_str_vis = f" ({time_part_match.group(1)}s)"
                                 break
                except Exception: pass

                final_msg_text = f"{self.selected_algorithm_text} Visualization Finished{time_str_vis}."
                final_msg_color = MSG_SUCCESS_COLOR if self.app_state == "SOLVED" else MSG_WARN_COLOR
                
                if self.app_state == "SOLVED":
                    final_msg_text += " Goal Reached!"
                else: # FAILED sau visualization
                    final_msg_text += " Goal NOT Reached."

                self.set_message(final_msg_text, final_msg_color)
                self.update_button_states()
                self.solution_path = None
                self.vis_step_index = 0
                self.final_state_after_vis = None

    def run_benchmark_threaded(self):
        if not self.user_start_state or not is_solvable(self.user_start_state): self.set_message("Cannot Benchmark: Board not ready/unsolvable.", MSG_ERROR_COLOR); return
        self.app_state = "BENCHMARKING"; self.update_button_states()
        self.set_message("Running Benchmark...", MSG_INFO_COLOR); self.set_results(["Benchmark Results:", "Running algorithms..."])
        self.draw(); pygame.display.flip(); start_state_copy = [r[:] for r in self.user_start_state]
        thread = threading.Thread(target=self._benchmark_thread_func, args=(start_state_copy, self.result_queue), daemon=True)
        thread.start()

    def _benchmark_thread_func(self, start_state_input, q):
        benchmark_results_list = []; total_time = 0; print("--- Starting Benchmark Run ---")
        for algo_func_bench in self.pathfinding_algos_for_benchmark:
            if algo_func_bench in self.algo_name_map:
                algo_txt_bench = self.algo_name_map[algo_func_bench]; print(f"Benchmarking: {algo_txt_bench}...")
                path_b, time_b, status_b, steps_b = None, 0, "Error", -1; t_start_b = time.perf_counter()
                try:
                    puzzle_b = Puzzle(start_state_input)
                    if algo_func_bench in ['iddfs', 'and_or_search']: puzzle_b.iddfs_max_depth = int(self.sliders['max_depth']['value'])
                    solver_b = getattr(puzzle_b, algo_func_bench, None)
                    if solver_b:
                        path_b = solver_b(); time_b = time.perf_counter() - t_start_b; total_time += time_b
                        if path_b and isinstance(path_b, list) and len(path_b) > 0:
                            is_goal = puzzle_b.is_goal(path_b[-1])
                            if is_goal: status_b = "Solved"; steps_b = len(path_b)-1 if self.buttons_config[algo_txt_bench]['type'] in ['path', 'path_if_found', 'beam_search'] else len(path_b)
                            else: status_b = "Finished"; steps_b = len(path_b)
                        else: status_b = "Failed/Stopped"
                    else: status_b = "Not Found"; time_b = time.perf_counter() - t_start_b; total_time += time_b
                except MemoryError: status_b = "Memory Error"; time_b = time.perf_counter() - t_start_b; total_time += time_b
                except Exception: status_b = "Runtime Error"; time_b = time.perf_counter() - t_start_b; total_time += time_b
                benchmark_results_list.append({'name': algo_txt_bench, 'time': time_b, 'steps': steps_b, 'status': status_b})
        print(f"--- Benchmark Finished. Total Time: {total_time:.3f}s ---")
        q.put({'status': 'benchmark_complete', 'message': f"Benchmark complete ({total_time:.2f}s)", 'results': benchmark_results_list, 'total_time': total_time, 'algo_name': 'benchmark', 'algo_text': 'Benchmark'})

    def _process_benchmark_result(self, result):
        all_res = result.get('results', []); total_t = result.get('total_time', 0)
        results_display = ["Benchmark Results:", "---------------------------------"]; all_res.sort(key=lambda x: x['name'])
        max_name_len = max(max(len(r['name']) for r in all_res), 12) if all_res else 12
        for res_item in all_res: # Renamed res to res_item
            t_str = f"{res_item['time']:.3f}s" if res_item['time'] > 0.0001 else "-"; name_s = res_item['name'].ljust(max_name_len)
            status_s = res_item['status']; steps_s = f"{res_item['steps']}" if res_item['steps'] != -1 else "N/A"
            steps_label = "Steps" if self.buttons_config.get(res_item['name'], {}).get('type', '') not in ['explore_path', 'generate_explore'] else "States"
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
                else: fastest = min(solved_path_res, key=lambda x: x['time']); sug = f"Suggestion: Fastest: {fastest['name']} ({fastest['steps']} steps, {fastest['time']:.3f}s)"
            except Exception: sug = "Suggestion: Error generating suggestion."
        results_display.append(sug); self.set_results(results_display)
        self.set_message(f"Benchmark complete ({total_t:.2f}s). See Results.", MSG_DEFAULT_COLOR)
        if self.user_start_state and is_solvable(self.user_start_state): self.app_state = "READY"
        elif self.app_state == "BENCHMARKING": self.app_state = "FAILED" # Or READY if preferred
        self.update_button_states()

    def update(self):
        if self.app_state == "PLACING_RANDOM": self._update_placement_animation()
        elif self.app_state in ["VISUALIZING", "VISUALIZING_BELIEF"]: self._update_visualization() # Combined check
        self._check_result_queue()

    def draw(self):
        self.screen.fill(LIGHT_GREY)
        pygame.draw.rect(self.screen, WHITE, self.controls_rect.inflate(10,10), border_radius=5)
        pygame.draw.rect(self.screen, WHITE, self.results_rect.inflate(4,4), border_radius=5)
        pygame.draw.rect(self.screen, LIGHT_GREY, self.bottom_panel_rect)
        pygame.draw.rect(self.screen, DARK_GREY, self.bottom_panel_rect, 1, border_radius=3)
        self.draw_grid(); self.draw_buttons(); self.draw_message();
        self.draw_results(); self.draw_sliders(); self.draw_notes()
        pygame.display.flip()

    def run(self):
        while self.running:
            self.handle_input(); self.update(); self.draw()
            self.clock.tick(60)
        pygame.quit()

# --- Main Execution ---
if __name__ == "__main__":
    try:
        app = PygamePuzzleApp()
        app.run()
    except Exception as main_exception:
        print("\n--- UNHANDLED EXCEPTION IN MAIN APPLICATION ---"); traceback.print_exc()
        pygame.quit(); sys.exit(1)