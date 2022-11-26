import tensorflow as tf
import numpy as np


def init():
    """
    A function to load pre-trained model into a global variable
    """
    
    global model
    # Loading model from trained artifact    
    model = tf.keras.models.load_model('./binaries/mnist.h5')


def predict(pixel_array: np.ndarray) -> dict:
    """
    A function to predict probabilities and assign digit to input image

    Args:
        pixel_array (np.ndarray): 28x28 numpy array (or List[List]), representing pixel
            values of a handwritten digit.
    
    Returns:
        (dict): Digit probabilities and most likely digit.
    """

    # Compute 10 probabilities, 1 for each possible digit
    predicted_probs = np.round(
        model.predict(np.array([pixel_array])).tolist()[0], 5
    )
    
    # Add the best possible matching digit to the output
    score = np.argmax(predicted_probs)
    
    return {
        "predicted_probs": predicted_probs.tolist(),
        "score": int(score)
    }

if __name__=="__main__":
    import json

    # Load saved model
    init()
    # Open sample input file and predict on each line/array
    with open("./data/sample_input.json", "r") as input:
        for record in input:
            pixel_array = json.loads(record)["pixel_array"]
            model_output = predict(pixel_array)
            print(json.dumps(model_output))