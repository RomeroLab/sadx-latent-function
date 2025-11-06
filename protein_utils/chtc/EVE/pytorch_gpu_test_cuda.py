import torch

print("Is Cuda available  :", torch.cuda.is_available())
print("Cuda device count  :", torch.cuda.device_count())
try:
  print("Cuda current device:", torch.cuda.current_device())
  print("Cuda device(0)     :", torch.cuda.device(0))
  print("Cuda device name   :", torch.cuda.get_device_name(0))
except:
  pass
