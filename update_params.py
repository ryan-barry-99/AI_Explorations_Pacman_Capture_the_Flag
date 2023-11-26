import json
import random

def update_parameters(param_json):
    """
    Update epsilon, alpha, and discount based on the total reward per episode.
    """
    # Load parameters from the JSON file
    with open(param_json, 'r') as file:
        params = json.load(file)

    # Extract current values
    epsilon = params["epsilon"][-1]
    alpha = params["alpha"][-1]
    discount = params["discount"][-1]

    reset_chance = 0.005  # Chance to reset parameters to initial values
    
    if len(params["total_reward"]) < 2 or random.random() < reset_chance:
        # Use initial values if there's not enough history
        epsilon = params["base"]["epsilon"]
        alpha = params["base"]["alpha"]
        discount = params["base"]["discount"]
    else:
        total_reward = params["total_reward"][-1]
        prev_reward = params["total_reward"][-2]

        # Update epsilon based on total reward
        if total_reward > prev_reward and total_reward > 0:
            epsilon *= 0.9  # Decrease epsilon if total reward is high
            alpha *= 0.9  # Decrease alpha if total reward is high
            discount *= 0.9  # Decrease discount if total reward is high
        elif total_reward < 0 or total_reward < prev_reward:
            epsilon *= 1.1  # Increase epsilon if total reward is low
            alpha *= 1.1  # Increase alpha if total reward is low
            discount *= 1.1  # Increase discount if total reward is low


        # Clip values to ensure they remain within valid ranges
        epsilon = max(0.0, min(1.0, epsilon))
        alpha = max(0.0, min(1.0, alpha))
        discount = max(0.0, min(1.0, discount))

    # Update the parameters dictionary
    params["epsilon"].append(epsilon)
    params["alpha"].append(alpha)
    params["discount"].append(discount)

    # Save the updated parameters back to the JSON file
    with open(param_json, 'w') as file:
        json.dump(params, file, indent=4)


if __name__ == "__main__":
    update_parameters("offensiveParams.json")
    update_parameters("defensiveParams.json")