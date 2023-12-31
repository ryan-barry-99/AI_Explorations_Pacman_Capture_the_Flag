import json

weights = {'closest-food': -2.2558226236802597, 
			'bias': 1.0856704846852672, 
		    '#-of-ghosts-1-step-away': -0.18419418670562, 
		    'successorScore': -0.027287497346388308, 
		    'eats-food': 9.970429654829946}
# Save weights to JSON file
with open('weights.json', 'w') as f:
    json.dump(weights, f)