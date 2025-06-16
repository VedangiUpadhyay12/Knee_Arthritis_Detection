import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

def predict():
    model = load_model("model_cnn.h5")
    img_size = 224
    base_path = "/content/auto_test"

    results = []

    for label in sorted(os.listdir(base_path)):
        folder = os.path.join(base_path, label)
        for file in os.listdir(folder):
            img_path = os.path.join(folder, file)
            img = image.load_img(img_path, target_size=(img_size, img_size))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0) / 255.0

            pred = np.argmax(model.predict(x))
            results.append({"Image": file, "True_Label": label, "Predicted_Label": pred})

    df = pd.DataFrame(results)
    df.to_csv("export_predictions.csv", index=False)
    print("✅ Predictions saved to export_predictions.csv")
