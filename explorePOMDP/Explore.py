# main.py
import os
import json
import random
import re
import datetime
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

from arenas import generate_map, grid_map_to_ascii

MOVE_DELTAS = {"N": (0, 1), "E": (1, 0), "S": (0, -1), "W": (-1, 0)}

# ==========================================
# 1. ENVIRONMENT CLASS (Absolute Actions, No Noise)
# ==========================================

class GridLocalizationEnv:
    def __init__(self, grid_map):
        self.grid_map = grid_map
        self.x = 0
        self.y = 0
        self.reset()

    def reset(self):
        """Spawns the robot at a random valid coordinate."""
        start_node = random.choice(list(self.grid_map.keys()))
        self.x, self.y = map(int, start_node.strip("()").split(","))
        return self._get_observation()

    def _get_observation(self):
        """Returns the exact item and global paths at the current node."""
        coord = f"({self.x},{self.y})"
        node_data = self.grid_map[coord]
        obs = {
            "item_here": node_data["item"],
            "paths_available": node_data["paths"]
        }
        return f"Observation: {json.dumps(obs)}"

    def step(self, action):
        """Executes absolute movements."""
        reward = -1  
        done = False
        is_success = False

        if action in ["move-n", "move-e", "move-s", "move-w"]:
            direction = action.split("-")[1].upper()
            coord = f"({self.x},{self.y})"
            
            # Move if path exists, otherwise bump wall and stay
            if direction in self.grid_map[coord]["paths"]:
                dx, dy = MOVE_DELTAS[direction]
                self.x += dx
                self.y += dy

        elif action.startswith("localize"):
            done = True
            match = re.search(r"localize\((.*?)\)", action)
            if match:
                guess = match.group(1).replace(" ", "")
                true_state = f"{self.x},{self.y}"
                if guess == true_state:
                    reward = 100
                    is_success = True
                else:
                    reward = -100
            else:
                reward = -100 
                
            return f"Episode Terminated. Guess: {action}. True State: ({self.x},{self.y}).", reward, done, is_success

        elif action == "wait":
            pass 
        else:
            return "Invalid Action.", reward, done, is_success

        return self._get_observation(), reward, done, is_success

# ==========================================
# 2. LLM UTILS & LOOP
# ==========================================

def get_system_prompt(map_dict):
    prompt = f"""You are a robot localizing yourself in a known 2D grid map.
Your starting coordinate is completely unknown. You have a built-in compass and know your absolute orientation at all times.

Here is the global map (Adjacency List):
{json.dumps(map_dict, indent=2)}

You will receive observations detailing the item at your current location and the absolute paths available (N, S, E, W).
Your available actions are: 'move-N', 'move-S', 'move-E', 'move-W', 'wait', and 'localize(x,y)'.
move-N means you attempt to move (0, +1) relative to your current location, 
move-S is (0, -1), 
move-E is (+1, 0), 
and move-W is (-1, 0).

CRITICAL INSTRUCTIONS:
1. Every turn, update a Candidate List of possible states formatted as (x,y).
2. Explicitly write out your logic for eliminating coordinates. Example: "I moved N. If I was at (1,1), I am now at (1,2). (1,2) has a Desk. My observation says 'None'. I eliminate (1,1) from my original list."
3. Your job is to reduce the maximum amount of uncertainty each turn. Therefore your movement decisions should be based on which paths will eliminate the most candidates in the next observation. Explain your reasoning step-by-step each turn.
3. When exactly 1 candidate remains, output: [ACTION: localize(x,y)]
4. Output your action strictly on a new line at the very end in this format: [ACTION: <your action>]."""
    return prompt

def clean_history(llm_response):
    return re.sub(r"<\|channel>thought.*?<channel\|>", "", llm_response, flags=re.DOTALL).strip()


def parse_action(llm_response):
    match = re.search(r"\[ACTION:\s*(move-n|move-s|move-e|move-w|wait|localize\([^)]+\))\]", llm_response, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return "invalid"

def run_localization_episode(grid_map, model_name="gemma-4-31b-it", max_steps=15, num_episodes=1, arena_name="5x5"):
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    
    folder = f"logs_explore_{arena_name}"
    os.makedirs(folder, exist_ok=True)
    
    for episode_num in range(1, num_episodes + 1):
        env = GridLocalizationEnv(grid_map)
        obs = env.reset()
        
        true_start = f"({env.x},{env.y})"
        print(f"--- Starting Episode {episode_num} ---")
        print(f"[SECRET] True Spawn: {true_start}")
        
        sys_prompt = get_system_prompt(grid_map)
        messages = [{"role": "user", "content": obs}]
        
        config = types.GenerateContentConfig(
            system_instruction=sys_prompt,
            temperature=1.0, top_p=0.95, top_k=64,
            thinking_config=types.ThinkingConfig(thinking_level="high")
        )

        total_reward = 0
        step = 0
        done = False
        
        while not done and step < max_steps:
            print(f"\nStep {step + 1} | {obs}")
            
            gemini_contents = []
            for msg in messages:
                role = "user" if msg["role"] == "user" else "model"
                gemini_contents.append({"role": role, "parts": [{"text": msg["content"]}]})

            try:
                response = client.models.generate_content(
                    model=model_name, contents=gemini_contents, config=config
                )
                llm_response = response.text
            except Exception as e:
                print(f"API Error: {e}")
                break
                
            print(f"Model Output:\n{llm_response}")
            messages.append({"role": "assistant", "content": clean_history(llm_response)})
            
            action = parse_action(llm_response)
            if action == "invalid":
                print("\n[!] Failed to parse action.")
                messages.append({"role": "user", "content": "Error: Format exactly as [ACTION: action_name]."})
                step += 1
                continue
                
            obs, reward, done, is_success = env.step(action)
            total_reward += reward
            messages.append({"role": "user", "content": obs})
            step += 1

        print("\n--- Episode Finished ---")
        print(f"Total Reward: {total_reward}")
        if step >= max_steps and not done:
            result = "Failed (Timeout)"
        elif done:
            result = "Success!" if is_success else "Catastrophic Failure"
        print(f"Result: {result}")
        
        # save
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(folder, f"episode_{episode_num}_{timestamp}.json")
        data = {
            "true_start": true_start,
            "total_reward": total_reward,
            "result": result,
            "arena": grid_map_to_ascii(grid_map),
            "messages": messages
        }
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)

if __name__ == "__main__":
   
    # Test on the 5x5 grid
    run_localization_episode(grid_map=generate_map(30, 30, wall_density=0), num_episodes=5, arena_name="30x30")