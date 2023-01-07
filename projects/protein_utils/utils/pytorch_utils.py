import torch

def get_accel_device():
    device = "cpu"
    try:
        if torch.backends.mps.is_available():
            device = "mps"
    except:
        pass

    try:
        if torch.cuda.is_available():
            device = "cuda"
    except:
        pass

    return device
