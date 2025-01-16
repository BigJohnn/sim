import pickle

with open('./logs/zeroth-walking/cfgs.pkl', 'rb') as f:
    data = pickle.load(f)
    print(data)
