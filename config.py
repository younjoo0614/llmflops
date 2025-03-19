import json 

with open('config.json', 'r') as f:
    config_data = json.load(f)

for key, value in config_data.items():
    globals()[key] = value