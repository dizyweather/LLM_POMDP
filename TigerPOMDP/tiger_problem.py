import os
import json
import random
import re
import requests
from datetime import datetime
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# 1. GLOBAL CONFIGURATION & MODULAR PROMPTS
# ==========================================

# Tiger POMDP Parameters
POMDP_CONFIG = {
    "reward_gold": 10,
    "penalty_tiger": -100,
    "cost_listen": -1,
    "listen_accuracy": 0.85,
    "max_steps": 10  # Max steps before forcing a reset/end
}

# Prompt Templates
PROMPTS = {
    "system_instruction": (
        "You are an agent solving the Tiger Problem. There are two doors: 'left' and 'right'. "
        "Your goal is to maximize your reward over the episode."
        f"Behind one door is a tiger (which will eat you, penalty {POMDP_CONFIG['penalty_tiger']}). Behind the other is gold (reward +{POMDP_CONFIG['reward_gold']}). "
        f"You can choose to 'listen' (cost {POMDP_CONFIG['cost_listen']}) to hear which door the tiger is behind, but the sound is only "
        f"{POMDP_CONFIG['listen_accuracy'] * 100}% accurate. However, you do get to choose again after listening. "
        "You can choose to open either door at any time, and get the corresponding reward of tiger vs treasure, but that will end the episode."
        "You could also choose to 'wait' and do nothing, which has no cost and you can choose again after."
        "Your available actions are: 'listen', 'open-left', 'open-right', 'wait'. "
        "CRITICAL INSTRUCTION: You must FIRST explain your step-by-step reasoning for your choice based on "
        "prior observations. THEN, you must output your chosen action strictly on a new line in this format: "
        "[ACTION: <your action>]."
    ),
    "initial_state": "The environment has started. You are facing the two doors. What is your action?",
    "observation_listen": "You listened and heard a tiger roar from the {direction}. What is your next action?",
    "observation_open": "You opened the {direction} door. {outcome}! Reward: {reward}. The environment will now reset."
}

# ==========================================
# 2. TIGER POMDP ENVIRONMENT
# ==========================================

class TigerPOMDP:
    def __init__(self, config):
        self.config = config
        self.tiger_loc = None
        self.reset()

    def reset(self):
        self.tiger_loc = random.choice(["left", "right"])
        return PROMPTS["initial_state"]

    def step(self, action):
        """Returns: observation_string, reward, done"""
        if action == "listen":
            # Noisy observation
            if random.random() < self.config["listen_accuracy"]:
                heard = self.tiger_loc
            else:
                heard = "right" if self.tiger_loc == "left" else "left"
            
            obs = PROMPTS["observation_listen"].format(direction=heard)
            return obs, self.config["cost_listen"], False

        elif action in ["open-left", "open-right"]:
            opened_door = action.split("-")[1]
            if opened_door == self.tiger_loc:
                outcome = "It was the tiger"
                reward = self.config["penalty_tiger"]
            else:
                outcome = "You found the gold"
                reward = self.config["reward_gold"]
            
            obs = PROMPTS["observation_open"].format(direction=opened_door, outcome=outcome, reward=reward)
            return obs, reward, True
            
        else:
            return "Invalid action. Please use 'listen', 'open-left', or 'open-right'.", 0, False

# ==========================================
# 3. LLM INTERFACE
# ==========================================

class LLMClient:
    def __init__(self, provider="ollama", model_name="llama3", gemini_api_key=None):
        self.provider = provider
        self.model_name = model_name
        
        if self.provider == "gemini":
            # Uses the new google.genai SDK
            if gemini_api_key:
                self.client = genai.Client(api_key=gemini_api_key)
            else:
                self.client = genai.Client() # Automatically falls back to GEMINI_API_KEY env variable

    def generate_response(self, messages):
        """Routes the request to the correct provider."""
        if self.provider == "ollama":
            return self._query_ollama(messages)
        elif self.provider == "gemini":
            return self._query_gemini(messages)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _query_ollama(self, messages):
        url = "http://localhost:11434/api/chat"
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False
        }
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()["message"]["content"]

    def _query_gemini(self, messages):
        gemini_contents = []
        system_instruction = None

        for msg in messages:
            if msg["role"] == "system":
                system_instruction = msg["content"]
                continue
            
            role = "user" if msg["role"] == "user" else "model"
            gemini_contents.append({"role": role, "parts": [{"text": msg["content"]}]})
        
        # Native Gemma 4 API configuration
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=1.0, 
            top_p=0.95,
            top_k=64,
            thinking_config=types.ThinkingConfig(thinking_level="high")
        ) if system_instruction else None
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=gemini_contents,
            config=config
        )
        return response.text

# ==========================================
# 4. INTERACTION LOOP & LOGGING
# ==========================================

def parse_action(llm_response):
    """Extracts the action from the LLM's text output."""
    if llm_response is None:
        print("Warning: LLM returned None. Defaulting to 'listen'.")
        return "invalid"
    
    match = re.search(r"\[ACTION:\s*(listen|open-left|open-right)\]", llm_response, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return "invalid"

def run_evaluation(provider="ollama", model_name="llama3", num_episodes=1):
    api_key = os.getenv("GEMINI_API_KEY")
    llm = LLMClient(provider=provider, model_name=model_name, gemini_api_key=api_key)
    env = TigerPOMDP(POMDP_CONFIG)
    average_reward = 0
    for episode in range(num_episodes):
        print(f"\n--- Starting Episode {episode + 1} ---")
        obs = env.reset()
        done = False
        step = 0
        total_reward = 0
        
        # Initialize conversation history for this specific episode
        messages = [
            {"role": "system", "content": PROMPTS["system_instruction"]},
            {"role": "user", "content": obs}
        ]
        
        while not done and step < POMDP_CONFIG["max_steps"]:
            print(f"Step {step + 1} | Env: {obs}")
            
            # 1. Get LLM Response
            try:
                llm_response = llm.generate_response(messages)
            except Exception as e:
                print(f"API Error: {e}")
                break
                
            print(f"Model:\n{llm_response}\n")
            messages.append({"role": "assistant", "content": llm_response})
            
            # 2. Parse Action
            action = parse_action(llm_response)
            if action == "invalid":
                print("Failed to parse action. Asking model to retry.")
                obs = "Error: Could not parse action. Remember to provide your reasoning, then format your action exactly as [ACTION: listen/open-left/open-right]."
                messages.append({"role": "user", "content": obs})
                step += 1
                continue
                
            # 3. Environment Step
            obs, reward, done = env.step(action)
            total_reward += reward
            messages.append({"role": "user", "content": obs})
            step += 1

        print(f"Episode {episode + 1} finished with Total Reward: {total_reward}")
        
        # Save messages to a distinct JSON file for this episode
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        foldername = f"logs_{provider}_{model_name}".replace(":", "_")
        os.makedirs(foldername, exist_ok=True)
        filename = os.path.join(foldername, f"episode_{episode + 1 + 10}_{timestamp}.json")
        
        episode_log = {
            "episode": episode + 1 + 10,
            "total_reward": total_reward,
            "steps_taken": step,
            "messages": messages
        }
        
        with open(filename, "w") as f:
            json.dump(episode_log, f, indent=4)
            
        print(f"Log for Episode {episode + 1 + 10} saved to {filename}")
    return average_reward / num_episodes if num_episodes > 0 else 0

if __name__ == "__main__":
    # Example usage:
    average_reward = run_evaluation(provider="gemini", model_name="gemma-4-26b-a4b-it", num_episodes=15)
    # average_reward = run_evaluation(provider="ollama", model_name="gemma4:e4b", num_episodes=15)

    print(f"\nAverage Reward over 10 episodes: {average_reward}")