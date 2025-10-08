from typing import List, Tuple, Dict, Iterable
from enum import Enum
import heapq

class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    FLASH = 4
    NUKE = 5

DIRS: List[Tuple[int, int, Action]] = [
    (-1, 0, Action.UP),
    (1, 0, Action.DOWN),
    (0, -1, Action.LEFT),
    (0, 1, Action.RIGHT),
]

def _extract(dct):
    rows, cols = dct["rows"], dct["cols"]
    obstacles = set(map(tuple, dct.get("obstacles", [])))
    creeps = dct.get("creeps", [])
    start = tuple(dct["start"])
    goals = set(map(tuple, dct.get("goals", [])))
    num_flash_left: int = dct.get("num_flash_left", 0)
    num_nuke_left: int = dct.get("num_nuke_left", 0)
    return rows, cols, obstacles, creeps, start, goals, num_flash_left, num_nuke_left

def _creeps_after_nukes(creep_map: Dict[Tuple[int,int], int],
                        r: int, c: int,
                        nuke_centers: Iterable[Tuple[int,int]]) -> int:
    # If within Manhattan 10 of any cast center -> cleared to 0
    for nr, nc in nuke_centers:
        if abs(nr - r) + abs(nc - c) <= 10:
            return 0
    return creep_map.get((r, c), 0)

def search(dct) -> list[int]:
    rows, cols, obstacles, creeps, start, goals, num_flash_left, num_nuke_left = _extract(dct)

    if not goals or start in obstacles or start in goals:
        return []

    creep_map: Dict[Tuple[int, int], int] = {(x, y): k for x, y, k in creeps}

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols

    start_state = (start[0], start[1], num_flash_left, num_nuke_left, frozenset())

    def h(state) -> int:
        r, c = state[0], state[1]
        return 2 * min(abs(r - gr) + abs(c - gc) for (gr, gc) in goals)

    g_cost: Dict[Tuple[int,int,int,int,frozenset], int] = {start_state: 0}
    parent: Dict[Tuple[int,int,int,int,frozenset], Tuple[int,int,int,int,frozenset]] = {}
    parent_actions: Dict[Tuple[int,int,int,int,frozenset], List[Action]] = {}

    open_heap: List[Tuple[int, int, Tuple[int,int,int,int,frozenset]]] = []
    tie = 0
    heapq.heappush(open_heap, (h(start_state), tie, start_state))

    while open_heap:
        _, _, cur = heapq.heappop(open_heap)
        cr, cc, cfl, cnk, centers = cur

        if (cr, cc) in goals:
            seq: List[Action] = []
            n = cur
            while n != start_state:
                acts = parent_actions[n]
                for a in reversed(acts):
                    seq.append(a)
                n = parent[n]
            seq.reverse()
            return [a.value for a in seq]

        cur_g = g_cost[cur]

        for dr, dc, move_act in DIRS:
            nr, nc = cr + dr, cc + dc
            if not in_bounds(nr, nc) or (nr, nc) in obstacles:
                continue
            effective_creeps = _creeps_after_nukes(creep_map, nr, nc, centers)
            step_cost = 4 + effective_creeps
            nxt = (nr, nc, cfl, cnk, centers)
            tentative = cur_g + step_cost
            if tentative < g_cost.get(nxt, float("inf")):
                g_cost[nxt] = tentative
                parent[nxt] = cur
                parent_actions[nxt] = [move_act]
                tie += 1
                heapq.heappush(open_heap, (tentative + h(nxt), tie, nxt))

        if cfl > 0:
            for dr, dc, move_act in DIRS:
                nr, nc = cr, cc
                path_cells: List[Tuple[int,int]] = []
                while True:
                    nr += dr
                    nc += dc
                    if not in_bounds(nr, nc) or (nr, nc) in obstacles:
                        break
                    path_cells.append((nr, nc))
                if not path_cells:
                    continue
                creep_loss = sum(_creeps_after_nukes(creep_map, r, c, centers)
                                 for (r, c) in path_cells)
                step_cost = 10 + 2 * len(path_cells) + creep_loss
                dest_r, dest_c = path_cells[-1]
                nxt = (dest_r, dest_c, cfl - 1, cnk, centers)
                tentative = cur_g + step_cost
                if tentative < g_cost.get(nxt, float("inf")):
                    g_cost[nxt] = tentative
                    parent[nxt] = cur
                    parent_actions[nxt] = [Action.FLASH, move_act]
                    tie += 1
                    heapq.heappush(open_heap, (tentative + h(nxt), tie, nxt))

        if cnk > 0 and (cr, cc) not in centers:
            new_centers = frozenset(set(centers) | {(cr, cc)})
            nxt = (cr, cc, cfl, cnk - 1, new_centers)
            step_cost = 50  # casting cost; position unchanged
            tentative = cur_g + step_cost
            if tentative < g_cost.get(nxt, float("inf")):
                g_cost[nxt] = tentative
                parent[nxt] = cur
                parent_actions[nxt] = [Action.NUKE]
                tie += 1
                heapq.heappush(open_heap, (tentative + h(nxt), tie, nxt))

    return []
