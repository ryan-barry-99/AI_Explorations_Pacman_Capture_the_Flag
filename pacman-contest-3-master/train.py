import subprocess
import time
from IPython.display import HTML, display, clear_output
import json
from tqdm import tqdm

offensive_params = json.load(open("offensiveParams.json", 'r'))
defensive_params = json.load(open("defensiveParams.json", 'r'))

num_episodes = offensive_params["num_episodes"]
total_episodes = 10000

# Function to display captured output
def show_output(output):
    display(HTML("<pre>" + output.decode("utf-8") + "</pre>"))


red = True
agents = ["myTeamApproxQLearningAgent", "myTeam2", "myTeam", "baselineTeam"]
# Function to run the capture.py command for a given episode
# Function to run the capture.py command for a given episode
def run_episode(episode):
    global red
    i = 0
    if red:
        command = f"python capture.py -r {agents[i]} -b myteam --delay-step 0 -q"
    else:
        command = f"python capture.py -r myteam -b {agents[i]} --delay-step 0 -q"
    if not red:
        i += 1
        if i == 4:
            i = 0
    # command = f"python capture.py -r myteam -b myteam --delay-step 0 -q"
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    if result.returncode != 0:
        error_message = result.stderr.decode("utf-8")
        print(f"Error in episode {episode + 1}: {error_message}")
    return result.returncode


# Run episodes
with tqdm(initial=num_episodes, total=total_episodes, desc=f"Training for {total_episodes} Episodes") as pbar:
    for episode in range(num_episodes, total_episodes):
        return_code = run_episode(episode)
        
        if return_code == 0:
            offensive_params = json.load(open("offensiveParams.json", 'r'))
            defensive_params = json.load(open("defensiveParams.json", 'r'))
            clear_output()
            red = not red
            if offensive_params["num_episodes"] % 100 == 0:
                subprocess.run("git add *")
                subprocess.run(f"git commit -m Trained for {offensive_params['num_episodes']} episodes")
                subprocess.run("git push")
        else:
            print(f"Error in episode {episode + 1}. Exiting.")
            break
        pbar.update(1)

print("All episodes completed.")
