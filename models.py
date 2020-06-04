import os
import sys
from tensorflow.keras.models import save_model, load_model


models_path = "models"
default_file = "latest"


def load_model_h5(name=default_file):
    name = os.path.join(models_path, name + ".h5")
    
    print("Loading model from", name)
    try:
        model = load_model(name)
    except:
        raise FileNotFoundError(f"File {name} not found")
        sys.exit()

    return model


def save_model_h5(model, name=default_file):
    if not os.path.isdir(models_path):
        os.makedirs(models_path)
    
    name = os.path.join(models_path, name + ".h5")
    print("Saving model to", name)
    save_model(model, name)
