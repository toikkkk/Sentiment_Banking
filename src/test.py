import torch
print(torch.cuda.is_available())  # Harus True
print(torch.cuda.device_count())  # Harus >= 1