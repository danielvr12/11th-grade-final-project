from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from DL3 import *
import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import seaborn as sns

model = DLModel("brain tumor classifier")
model.add(DLLayer("Layer 1",  1024, (3072,), activation="relu", W_initialization="He", learning_rate=0.02, random_scale=0.01))
model.add(DLLayer("Layer 2",  512, (1024,), activation="relu", W_initialization="He", learning_rate=0.2, random_scale=0.01))
model.add(DLLayer("Layer 3",  256, (512,), activation="tanh", W_initialization="He", learning_rate=0.1, random_scale=0.01))
model.add(DLLayer("Layer 4",  4, (256,), activation="trim_softmax", W_initialization="He", learning_rate=0.02, random_scale=0.01))
model.compile("categorical_cross_entropy")

activations = ["relu", "relu", "tanh", "trim_softmax"] 
loss_function = "categorical_cross_entropy" 
model.load_weights("saved_weights 80.53%", activations, loss_function)

def prepare_image(file_path, image_size=(32, 32)):
    img = Image.open(file_path)
    img = img.resize(image_size)
    img_array = np.array(img).reshape(image_size[0]*image_size[1]*3,) / 255.0 - 0.5
    return img_array.reshape(-1, 1)  # Reshape for the model input

classNames = ["pituitary_tumor","no_tumor","meningioma_tumor","glioma_tumor"]
image_path = "scan35.jpg"  # Replace with the actual path to your image
image_data = prepare_image(image_path)
prediction = model.predict(image_data)
predicted_class = np.argmax(prediction, axis=0)
print("Predicted class:", classNames[predicted_class[0]])


