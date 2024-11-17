import pickle
from pprint import pprint

with open('./logs/zeroth-walking/cfgs.pkl', 'rb') as f:
    data = pickle.load(f)
    pprint(data, indent=2, width=80, depth=3)
