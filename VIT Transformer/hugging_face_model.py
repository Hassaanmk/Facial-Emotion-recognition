from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import matplotlib.pyplot as plt
import torch

processor = AutoImageProcessor.from_pretrained("motheecreator/vit-Facial-Expression-Recognition")
model = AutoModelForImageClassification.from_pretrained("motheecreator/vit-Facial-Expression-Recognition")

# Load an image (replace the path with your image file)
image = Image.open("C:/Users/PC/Downloads/happy.jpg")
inputs = processor(images=image, return_tensors="pt")

# Make predictions
with torch.no_grad():
    outputs = model(**inputs)

# Get predicted class index and label
logits = outputs.logits
predicted_class_idx = logits.argmax(-1).item()
labels = model.config.id2label
predicted_label = labels[predicted_class_idx]

# Output the result
print(f"Predicted label: {predicted_label}")

# Display the image with its predicted label
plt.imshow(image)
plt.title(f"Predicted: {predicted_label}")
plt.axis("off")
plt.show()

