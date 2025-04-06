import os

def get_data_path():
    with open(os.path.join(os.path.dirname(__file__), 'data_path.txt'), 'r') as f:
        return f.read().strip()