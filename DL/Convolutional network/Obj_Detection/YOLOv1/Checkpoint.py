import os
import pickle
from Yolov1_Resnet50 import ResNet50

def Load_model(model_path):
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            print(f'Loaded model from {model_path} successfully !!!')
            return model
    except Exception as e:
        print(f'Error loading model from {model_path} with error: {e}')
        model = ResNet50()
        print(f'Initialized new model successfully !!!')
        return model

def Save_model(model, path):
    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)
    print("here")
    try:
        with open(path, 'wb') as f:
            pickle.dump(model, f)
            print(f'Saved model to {path} successfully !!!')
    except Exception as e:
        print(f'Error saving model to {path} with error: {e}')

def restore_lr(model, new_lr):
    count = 0
    def recursive_reset(obj):
        nonlocal count
        if hasattr(obj, 'lr'):
            obj.lr = new_lr
            count += 1

        if hasattr(obj, '__dict__'):
            for key, value in obj.__dict__.items():
                if hasattr(value, '__dict__'):
                    recursive_reset(value)
                elif isinstance(value, list):
                    for item in value:
                        recursive_reset(item)

    recursive_reset(model)
    print(f'Restored learning rates = {new_lr} for {count} layers successfully !!!')
