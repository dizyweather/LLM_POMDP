# for each json file in target folder, read the total reward and print the average total reward across all files
import os
import json

def calculate_average_reward(foldername):
    
    total_rewards = 0
    total_episode_length = 0
    total_episode = 0
    
    for filename in os.listdir(foldername):
        if filename.endswith(".json"):
            with open(os.path.join(foldername, filename), "r") as f:
                data = json.load(f)
                total_rewards += data["total_reward"]
                total_episode_length += data["episode"]  # Assuming episode numbers are sequential and start from 1
                total_episode += 1

    if total_rewards:
        average_reward = total_rewards / total_episode
        average_episode = total_episode_length / total_episode
        print(f"Average Total Reward across {total_episode} episodes: {average_reward:.2f}")
        print(f"Average Total Episodes across {total_episode} episodes: {average_episode:.2f}")
    else:
        print("No JSON files found in the folder.")

if __name__ == "__main__":
    foldername = "TigerPOMDP/logs_gemini_gemma-4-31b-it"
    print(calculate_average_reward(foldername))
