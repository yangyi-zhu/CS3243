from typing import List, Tuple, Dict
from collections import deque
import heapq

Coord = Tuple[int, int]

# --- Helper ---

# Returns list of neighboring cells to move to
def _neighbors(pos: Coord, rows: int, cols: int, obstacles: set) -> List[Coord]:
    r, c = pos
    candidates = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
    return [
        (nr, nc)
        for nr, nc in candidates
        if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in obstacles
    ]

# Reconstruct path after goal found
def _build_path(parent: Dict[Coord, Coord], start: Coord, goal: Coord) -> List[Coord]:
    path = [goal]
    cur = goal
    while cur != start:
        cur = parent[cur]
        path.append(cur)
    return path[::-1]

# Extract information from dict
def _extract(dct):
    rows, cols = dct["rows"], dct["cols"]
    obstacles = set(map(tuple, dct["obstacles"]))
    start = tuple(dct["start"])
    goals = set(map(tuple, dct["goals"]))
    return rows, cols, obstacles, start, goals

# --- Search ---

def dfs_search(dct) -> List[Tuple[int, int]]:
    rows, cols, obstacles, start, goals = _extract(dct)
    if start in obstacles:
        return []
    if start in goals:
        return [start]

    stack = [start]
    visited = {start}
    parent = {}

    while stack:
        cur = stack.pop()
        if cur in goals:
            return _build_path(parent, start, cur)
        for nb in reversed(_neighbors(cur, rows, cols, obstacles)):
            if nb not in visited:
                visited.add(nb)
                parent[nb] = cur
                stack.append(nb)
    return []


def bfs_search(dct) -> List[Tuple[int, int]]:
    rows, cols, obstacles, start, goals = _extract(dct)
    if start in obstacles:
        return []
    if start in goals:
        return [start]

    q = deque([start])
    visited = {start}
    parent: Dict[Tuple[int, int], Tuple[int, int]] = {}

    while q:
        cur = q.popleft()
        for nb in _neighbors(cur, rows, cols, obstacles):
            if nb not in visited:
                visited.add(nb)
                parent[nb] = cur

                if nb in goals:
                    return _build_path(parent, start, nb)

                q.append(nb)

    return []


def ucs_search(dct) -> List[Tuple[int, int]]:
    rows, cols, obstacles, start, goals = _extract(dct)
    if start in obstacles:
        return []
    if start in goals:
        return [start]

    pq = [(0, start)]
    cost = {start: 0}
    parent = {}

    while pq:
        g, cur = heapq.heappop(pq)
        if cur in goals:
            return _build_path(parent, start, cur)
        if g > cost[cur]:
            continue
        for nb in _neighbors(cur, rows, cols, obstacles):
            new_cost = g + 1
            if new_cost < cost.get(nb, float("inf")):
                cost[nb] = new_cost
                parent[nb] = cur
                heapq.heappush(pq, (new_cost, nb))
    return []