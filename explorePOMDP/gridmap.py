# arenas.py

def build_grid_map(width, height, walls, items):
    """
    Generates an adjacency list dictionary for a grid of size (width x height).
    walls: list of tuples containing pairs of blocked coordinates e.g., [((0,0), (0,1))]
    items: dict mapping string coordinates to item names e.g., {"(2,2)": "desk"}
    """
    grid = {}
    
    # Helper to check if a move is blocked by a wall
    def is_blocked(c1, c2):
        return (c1, c2) in walls or (c2, c1) in walls

    for x in range(width):
        for y in range(height):
            coord = f"({x},{y})"
            paths = []
            
            # Check all four cardinal directions for boundaries and walls
            if y + 1 < height and not is_blocked((x, y), (x, y + 1)): paths.append("N")
            if x + 1 < width and not is_blocked((x, y), (x + 1, y)): paths.append("E")
            if y - 1 >= 0 and not is_blocked((x, y), (x, y - 1)): paths.append("S")
            if x - 1 >= 0 and not is_blocked((x, y), (x - 1, y)): paths.append("W")
            
            grid[coord] = {
                "item": items.get(coord, "none"),
                "paths": paths
            }
    return grid

def print_ascii_map(grid, width, height):
    """Generates a visual representation of the generated grid map."""
    print(f"\n--- Map Visualization ({width}x{height}) ---")
    for y in range(height - 1, -1, -1):
        row_top = ""
        row_mid = ""
        for x in range(width):
            coord = f"({x},{y})"
            node = grid[coord]
            
            # Item formatting
            item = node["item"]
            item_str = f"[{item[:4].capitalize()}]" if item != "none" else "[    ]"
            
            # Paths
            east_path = "---" if "E" in node["paths"] else "   "
            row_top += item_str + east_path
            
            if "S" in node["paths"]:
                row_mid += "  |       "
            else:
                row_mid += "          "
                
        print(row_top)
        if y > 0:
            print(row_mid)
    print("----------------------------------\n")


# ==========================================
# PRE-DEFINED ARENAS
# ==========================================

# 1. THE 3x3 ARENA (From previous example)
walls_3x3 = [((0,1), (0,2)), ((1,2), (2,2)), ((0,0), (0,1))] # Removed some paths to match previous
items_3x3 = {"(0,2)": "sink", "(2,2)": "desk", "(1,1)": "trashcan", "(2,0)": "sink"}
MAP_3X3 = build_grid_map(3, 3, walls_3x3, items_3x3)

# 2. THE 5x5 ARENA
# Creating some rooms and hallways
walls_5x5 = [
    ((1,1), (1,2)), ((2,1), (2,2)), ((3,1), (3,2)), # Horizontal wall segment
    ((2,3), (3,3)), ((2,4), (3,4))                  # Vertical wall segment
]
items_5x5 = {
    "(0,0)": "trashcan", "(4,0)": "desk", 
    "(0,4)": "sink", "(4,4)": "trashcan",
    "(2,2)": "plant"
}
MAP_5X5 = build_grid_map(5, 5, walls_5x5, items_5x5)

# 3. THE 10x10 ARENA (Sparse Environment)
walls_10x10 = [
    ((2,2), (2,3)), ((2,3), (2,4)), ((2,4), (2,5)), # Long vertical wall
    ((7,7), (8,7)), ((8,7), (9,7)),                 # Short horizontal wall
    ((5,5), (5,6)), ((6,5), (6,6))                  # Small central block
]
items_10x10 = {
    "(1,1)": "desk", "(8,2)": "trashcan", "(2,8)": "sink",
    "(9,9)": "desk", "(5,1)": "plant", "(4,8)": "trashcan",
    "(7,5)": "sink"
}
MAP_10X10 = build_grid_map(10, 10, walls_10x10, items_10x10)

if __name__ == "__main__":
    # Test the visualizations by running `python arenas.py`
    print_ascii_map(MAP_3X3, 3, 3)
    print_ascii_map(MAP_5X5, 5, 5)
    print_ascii_map(MAP_10X10, 10, 10)