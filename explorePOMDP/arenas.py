# arenas.py
import random

def generate_items(seed=42, width=10, height=10):
    random.seed(seed)
    # Create exactly 25 of each category (100 total)
    item_pool = {"none": 0.5, "desk": 0.2, "plant":0.1, "trashcan": 0.2}
    
    # Map them to the 10x10 coordinates
    items = {}
    
    for y in range(height):
        for x in range(width):
            item = random.choices(list(item_pool.keys()), weights=item_pool.values(), k=1)[0]
            items[f"({x},{y})"] = item
            
    return items

# Generate the items
def build_grid_map(width, height, walls, items):
    """
    Generates an adjacency list dictionary for a grid of size (width x height).
    walls: list of tuples containing pairs of blocked coordinates e.g., [((0,0), (0,1))]
    items: dict mapping string coordinates to item names e.g., {"(2,2)": "desk"}
    """
    grid = {}
    
    def is_blocked(c1, c2):
        return (c1, c2) in walls or (c2, c1) in walls

    for x in range(width):
        for y in range(height):
            coord = f"({x},{y})"
            paths = []
            
            if y + 1 < height and not is_blocked((x, y), (x, y + 1)): paths.append("N")
            if x + 1 < width and not is_blocked((x, y), (x + 1, y)): paths.append("E")
            if y - 1 >= 0 and not is_blocked((x, y), (x, y - 1)): paths.append("S")
            if x - 1 >= 0 and not is_blocked((x, y), (x - 1, y)): paths.append("W")
            
            grid[coord] = {
                "item": items.get(coord, "none"),
                "paths": paths
            }
    return grid

def grid_map_to_ascii(grid, width=None, height=None):
    """Return a visual ASCII representation of the grid in coordinate[Item] format."""
    coords = [tuple(map(int, c.strip("()").split(","))) for c in grid.keys()]
    xs = [x for x, y in coords]
    ys = [y for x, y in coords]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    if width is None:
        width = max_x - min_x + 1
    if height is None:
        height = max_y - min_y + 1

    lines = [f"--- Map Visualization ({width}x{height}) ---"]
    for y in range(max_y, min_y - 1, -1):
        row_top = ""
        row_mid = ""
        for x in range(min_x, max_x + 1):
            coord = f"({x},{y})"
            node = grid[coord]
            
            # Format item: Capitalize and truncate to 5 chars (e.g., Trash)
            item = node["item"].capitalize()[:5]
            cell_str = f"{coord}[{item}]"
            
            # Pad the string to align columns cleanly
            padded_cell = f"{cell_str:<13}"
            
            if "E" in node["paths"]:
                row_top += padded_cell + "--- "
            else:
                row_top += padded_cell + "    "
                
            if y > min_y:
                if "S" in node["paths"]:
                    row_mid += "  |              "
                else:
                    row_mid += "                 "
                
        lines.append(row_top.rstrip())
        if y > min_y:
            lines.append(row_mid.rstrip())
    lines.append("----------------------------------")
    return "\n".join(lines)


def print_ascii_map(grid, width, height):
    """Print a visual ASCII representation of the grid."""
    print(grid_map_to_ascii(grid, width, height) + "\n")


# ==========================================
# PRE-DEFINED ARENAS
# ==========================================

# 1. THE 3x3 ARENA 
# walls_3x3 = [((0,1), (0,2)), ((1,2), (2,2)), ((0,0), (0,1))]
walls_3x3 = []
items_3x3 = generate_items(seed=0, width=3, height=3)
MAP_3X3 = build_grid_map(3, 3, walls_3x3, items_3x3)

# --- Map Visualization (3x3) ---
# (0,2)[Sink]  --- (1,2)[None]  --- (2,2)[Desk]
#   |                |                |
# (0,1)[None]  --- (1,1)[Trash] --- (2,1)[None]
#   |                |                |
# (0,0)[None]  --- (1,0)[None]  --- (2,0)[Trash]

# 2. THE 5x5 ARENA
walls_5x5 = [
    # ((1,1), (1,2)), ((2,1), (2,2)), ((3,1), (3,2)), 
    # ((2,3), (3,3)), ((2,4), (3,4))                  
]
items_5x5 = generate_items(seed=0, width=5, height=5)
MAP_5X5 = build_grid_map(5, 5, walls_5x5, items_5x5)

# --- Map Visualization (5x5) ---
# (0,4)[Sink]  --- (1,4)[None]  --- (2,4)[None]  --- (3,4)[None]  --- (4,4)[Trash]
#   |                |                |                |                |
# (0,3)[None]  --- (1,3)[Plant]  --- (2,3)[None]  --- (3,3)[None]  --- (4,3)[None]
#   |                |                |                |                |
# (0,2)[None]  --- (1,2)[None]  --- (2,2)[Plant] --- (3,2)[Trash]  --- (4,2)[None]
#   |                                                                   |
# (0,1)[None]  --- (1,1)[Trash]  --- (2,1)[None]  --- (3,1)[None]  --- (4,1)[None]
#   |                |                |                |                |
# (0,0)[Trash] --- (1,0)[None]  --- (2,0)[None]  --- (3,0)[None]  --- (4,0)[Desk]
# ----------------------------------

# 3. THE 10x10 ARENA
walls_10x10 = [
    ((2,2), (2,3)), ((2,3), (2,4)), ((2,4), (2,5)), 
    ((7,7), (8,7)), ((8,7), (9,7)),                 
    ((5,5), (5,6)), ((6,5), (6,6))                  
]

items_10x10 = generate_items(seed=0, width=10, height=10)
MAP_10X10 = build_grid_map(10, 10, walls_10x10, items_10x10)

def generate_map(width, height, wall_density=0.1, seed=42):
    random.seed(seed)
    walls = []
    for x in range(width):
        for y in range(height):
            if random.random() < wall_density:
                # Randomly block one of the possible paths from this cell
                direction = random.choice(["N", "E", "S", "W"])
                if direction == "N" and y + 1 < height:
                    walls.append(((x, y), (x, y + 1)))
                elif direction == "E" and x + 1 < width:
                    walls.append(((x, y), (x + 1, y)))
                elif direction == "S" and y - 1 >= 0:
                    walls.append(((x, y), (x, y - 1)))
                elif direction == "W" and x - 1 >= 0:
                    walls.append(((x, y), (x - 1, y)))
    items = generate_items(seed=seed, width=width, height=height)
    return build_grid_map(width, height, walls, items)
if __name__ == "__main__":
    # print_ascii_map(MAP_3X3, 3, 3)
    # print_ascii_map(MAP_5X5, 5, 5)
    print_ascii_map(MAP_10X10, 10, 10)