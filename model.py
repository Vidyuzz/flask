import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

class SimpleMNISTModel:
    def __init__(self):
        # Load & preprocess
        (x_train, y_train), _ = mnist.load_data()
        x_train = x_train.astype('float32') / 255.0
        y_train = to_categorical(y_train, 10)

        # Build a tiny network
        self.model = Sequential([
            Flatten(input_shape=(28, 28)),
            Dense(128, activation='relu'),
            Dense(10, activation='softmax'),
        ])
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Train briefly (3 epochs)
        self.model.fit(x_train, y_train, epochs=3, batch_size=32, validation_split=0.1)

    def predict(self, img_array: np.ndarray) -> int:
        # Normalize & reshape
        img = img_array.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)  # shape (1,28,28)
        preds = self.model.predict(img)
        return int(np.argmax(preds, axis=1)[0])

# Instantiate once at import
_simple_model = SimpleMNISTModel()

def predict_digit(image_array: np.ndarray) -> int:
    """
    image_array: 28×28 numpy array (grayscale)
    returns: predicted digit 0–9
    """
    return _simple_model.predict(image_array)
