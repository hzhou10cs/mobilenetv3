import json
import numpy as np

file_path = 'tx2_inf_latency.json'

with open(file_path, 'r') as file:
    record:dict = json.load(file)

data = np.array([[item_value for item_value in trace.values()] for trace in record.values()])

np.savetxt('monitoring_data.txt', data)