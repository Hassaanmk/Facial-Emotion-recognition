# Use a pipeline as a high-level helper
# Load model directly
from transformers import AutoImageProcessor, AutoModelForImageClassification

processor = AutoImageProcessor.from_pretrained("motheecreator/vit-Facial-Expression-Recognition")
model = AutoModelForImageClassification.from_pretrained("motheecreator/vit-Facial-Expression-Recognition")
import torch # type: ignore
print(torch.cuda.is_available())  # This should return True if a compatible GPU is found

import torch
print(torch.cuda.is_available())  # Should return True
print(torch.cuda.current_device())  # Should return 0 if the GPU is recognized
print(torch.cuda.get_device_name(0))  # Should return the name of your GPU
