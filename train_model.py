from cnn_model import build_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def train_model():
    img_size = 224
    batch_size = 32

    train_path = '/content/train'
    val_path = '/content/val'

    datagen = ImageDataGenerator(rescale=1./255)

    train = datagen.flow_from_directory(train_path,
                                        target_size=(img_size, img_size),
                                        class_mode='categorical',
                                        batch_size=batch_size)

    val = datagen.flow_from_directory(val_path,
                                      target_size=(img_size, img_size),
                                      class_mode='categorical',
                                      batch_size=batch_size)

    model = build_model()
    model.fit(train, validation_data=val, epochs=10)

    model.save("model_cnn.h5")
    print("✅ Model trained and saved as model_cnn.h5")
