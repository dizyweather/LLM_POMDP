import os
import json
import random
import re
import datetime
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()


# ==========================================
# 1. ENVIRONMENT CONFIGURATION & MAP
# ==========================================

# A simple 3x3 grid with walls and sparse items.
# (x, y) format. 0,0 is bottom left.
# Map topology:
# (0,2)[Sink] --- (1,2)[None]     (2,2)[Desk]
#   |               |               |
# (0,1)[None]     (1,1)[Trash]--- (2,1)[None]
#   |               |               |
# (0,0)[None] --- (1,0)[None] --- (2,0)[Sink]

GLOBAL_MAP = {
    "(0,2)": {"item": "sink", "paths": ["S", "E"]},
    "(1,2)": {"item": "none", "paths": ["S", "W"]},
    "(2,2)": {"item": "desk", "paths": ["S"]},
    "(0,1)": {"item": "none", "paths": ["N", "S"]},
    "(1,1)": {"item": "trashcan", "paths": ["N", "S", "E"]},
    "(2,1)": {"item": "none", "paths": ["N", "S", "W"]},
    "(0,0)": {"item": "none", "paths": ["N", "E"]},
    "(1,0)": {"item": "none", "paths": ["N", "E", "W"]},
    "(2,0)": {"item": "sink", "paths": ["N", "W"]}
}

DIRECTIONS = ["N", "E", "S", "W"]

# Directional translations
MOVE_DELTAS = {"N": (0, 1), "E": (1, 0), "S": (0, -1), "W": (-1, 0)}

# Relative mapping based on current heading
RELATIVE_TO_GLOBAL = {
    "N": {"front": "N", "back": "S", "left": "W", "right": "E"},
    "E": {"front": "E", "back": "W", "left": "N", "right": "S"},
    "S": {"front": "S", "back": "N", "left": "E", "right": "W"},
    "W": {"front": "W", "back": "E", "left": "S", "right": "N"}
}

# ==========================================
# 2. PROMPT TEMPLATES
# ==========================================

PROMPTS = {
    "system_instruction": f"""You are a robot trying to localize yourself in a known 2D grid map.
Your starting coordinate and orientation are completely unknown.

Here is the global map of the environment (Adjacency List):
{json.dumps(GLOBAL_MAP, indent=2)}

You will receive observations in RELATIVE directions (front, back, left, right).
Your available actions are: 'move-forward', 'turn-left', 'turn-right', 'wait', and 'localize(x,y,facing)'.

CRITICAL INSTRUCTIONS:
1. Every turn, you must maintain and update a Candidate List of possible states formatted as (x, y, facing).
2. Before choosing an action, explicitly write out your coordinate translation. For example: "If I was at (1,1, E) and turn-left, I am now at (1,1, N)."
3. Cross-reference your new hypothetical states with the global map and the new relative observation. Eliminate candidates that contradict the map.
4. When exactly 1 candidate remains, you must use the localize action: [ACTION: localize(x,y,facing)]
5. Output your chosen action strictly on a new line at the very end in this format: [ACTION: <your action>]."""
}

# ==========================================
# 3. ENVIRONMENT CLASS
# ==========================================

class GridLocalizationEnv:
    def __init__(self, grid_map):
        self.grid_map = grid_map
        self.x = 0
        self.y = 0
        self.orientation = "N"
        self.reset()

    def reset(self):
        """Spawns the robot at a random valid coordinate with a random orientation."""
        start_node = random.choice(list(self.grid_map.keys()))
        # Parse "(x,y)" string
        self.x, self.y = map(int, start_node.strip("()").split(","))
        self.orientation = random.choice(DIRECTIONS)
        return self._get_observation()

    def _get_observation(self):
        """Translates global map data at current pos into relative sensor readings."""
        coord = f"({self.x},{self.y})"
        node_data = self.grid_map[coord]
        global_paths = node_data["paths"]
        
        # Translate global paths to relative paths
        relative_paths = []
        for rel, glob in RELATIVE_TO_GLOBAL[self.orientation].items():
            if glob in global_paths:
                relative_paths.append(rel)
                
        obs = {
            "item_here": node_data["item"],
            "paths_available": relative_paths
        }
        return f"Observation: {json.dumps(obs)}"

    def step(self, action):
        """Executes action, returns (observation, reward, done, is_success)"""
        reward = -1  # Standard exploration step cost
        done = False
        is_success = False

        if action == "move-forward":
            # Check if there is a path in the direction we are facing
            coord = f"({self.x},{self.y})"
            if self.orientation in self.grid_map[coord]["paths"]:
                dx, dy = MOVE_DELTAS[self.orientation]
                self.x += dx
                self.y += dy
            # If no path, we just bump into a wall and stay in place

        elif action == "turn-left":
            idx = DIRECTIONS.index(self.orientation)
            self.orientation = DIRECTIONS[(idx - 1) % 4]

        elif action == "turn-right":
            idx = DIRECTIONS.index(self.orientation)
            self.orientation = DIRECTIONS[(idx + 1) % 4]

        elif action.startswith("localize"):
            done = True
            # Parse localize(x,y,facing)
            match = re.search(r"localize\((.*?)\)", action)
            if match:
                guess = match.group(1).replace(" ", "")
                true_state = f"{self.x},{self.y},{self.orientation}"
                if guess == true_state:
                    reward = 100
                    is_success = True
                else:
                    reward = -100
            else:
                reward = -100 # Failed to format the localize action properly
                
            return f"Episode Terminated. Guess: {action}. True State: ({self.x},{self.y},{self.orientation}).", reward, done, is_success

        elif action == "wait":
            pass # Do nothing
            
        else:
            return "Invalid Action.", reward, done, is_success

        return self._get_observation(), reward, done, is_success

# ==========================================
# 4. UTILITIES & API CLIENT
# ==========================================

def clean_history(llm_response):
    """Removes Gemma 4's thought blocks to prevent context pollution."""
    return re.sub(r"<\|channel>thought.*?<channel\|>", "", llm_response, flags=re.DOTALL).strip()

def parse_action(llm_response):
    """Extracts the action from the LLM's text output."""
    match = re.search(r"\[ACTION:\s*(move-forward|turn-left|turn-right|wait|localize\([^)]+\))\]", llm_response, re.IGNORECASE)
    if match:
        return match.group(1)
    return "invalid"

# ==========================================
# 5. MAIN LOOP
# ==========================================

def run_localization_episode(model_name="gemma-4-31b-it", max_steps=15):
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    env = GridLocalizationEnv(GLOBAL_MAP)
    
    obs = env.reset()
    true_start = f"({env.x},{env.y},{env.orientation})"
    print(f"--- Starting Episode ---")
    print(f"[SECRET] True Spawn: {true_start}")
    
    messages = [
        {"role": "user", "content": obs}
    ]
    
    # Configure Gemini API for deep reasoning
    config = types.GenerateContentConfig(
        system_instruction=PROMPTS["system_instruction"],
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        thinking_config=types.ThinkingConfig(thinking_level="high")
    )

    total_reward = 0
    step = 0
    done = False
    
    while not done and step < max_steps:
        print(f"\nStep {step + 1} | {obs}")
        
        # Prepare history for Gemini API
        gemini_contents = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            gemini_contents.append({"role": role, "parts": [{"text": msg["content"]}]})
            
        # Call API
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=gemini_contents,
                config=config
            )
            llm_response = response.text
        except Exception as e:
            print(f"API Error: {e}")
            break
            
        print(f"Model Output:\n{llm_response}")
        
        # Clean the response and add to history
        cleaned_response = clean_history(llm_response)
        messages.append({"role": "assistant", "content": cleaned_response})
        
        # Parse and execute action
        action = parse_action(llm_response)
        if action == "invalid":
            print("\n[!] Failed to parse action. Asking model to retry.")
            obs = "Error: Could not parse action. Ensure it is formatted exactly as [ACTION: action_name]."
            messages.append({"role": "user", "content": obs})
            step += 1
            continue
            
        print(f"\n[Environment] Executing: {action}")
        obs, reward, done, is_success = env.step(action)
        total_reward += reward
        messages.append({"role": "user", "content": obs})
        step += 1

    # Save conversation to JSON file
    folder = "logs_explore"
    os.makedirs(folder, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{folder}/episode_{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump(messages, f, indent=2)
    print(f"Conversation saved to {filename}")

    print("\n--- Episode Finished ---")
    print(f"Total Reward: {total_reward}")
    if step >= max_steps and not done:
        print("Result: Failed (Timeout)")
    elif done:
        print(f"Result: {'Success!' if is_success else 'Catastrophic Failure (Wrong Location)'}")
        print(f"End State Output: {obs}")

if __name__ == "__main__":
    # If testing locally, ensure GEMINI_API_KEY is exported in your terminal.
    run_localization_episode(model_name="gemma-4-31b-it")