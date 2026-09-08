import os
import random
import numpy as np
import matplotlib.pyplot as plt

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass

def get_project_paths(base_path="/content/drive/MyDrive/Bot_Detection_Project"):
    """
    Returns path dictionary with Google Drive base path if available,
    or falls back to current local directory structure.
    """
    if not os.path.exists("/content/drive/MyDrive"):
        base_path = os.path.abspath(".")
        
    paths = {
        'BASE': base_path,
        'DATA_INSTAGRAM': os.path.join(base_path, 'data', 'instagram'),
        'DATA_TWITTER': os.path.join(base_path, 'data', 'twitter'),
        'OUTPUT_INSTAGRAM_GRAPHS': os.path.join(base_path, 'outputs', 'instagram', 'graphs'),
        'OUTPUT_INSTAGRAM_MODELS': os.path.join(base_path, 'outputs', 'instagram', 'models'),
        'OUTPUT_INSTAGRAM_RESULTS': os.path.join(base_path, 'outputs', 'instagram', 'results'),
        'OUTPUT_TWITTER_GRAPHS': os.path.join(base_path, 'outputs', 'twitter', 'graphs'),
        'OUTPUT_TWITTER_MODELS': os.path.join(base_path, 'outputs', 'twitter', 'models'),
        'OUTPUT_TWITTER_RESULTS': os.path.join(base_path, 'outputs', 'twitter', 'results')
    }
    
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
        
    return paths
