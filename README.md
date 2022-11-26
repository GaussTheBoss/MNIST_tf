# MNIST_tf
A sample model to demonstrate multi-class classification using **tensorflow**. The saved model was trained on the MNIST dataset.

## Running Locally

To run this model locally, create a new Python 3.8.3 virtual environment
(such as with `pyenv`). Then, use the following command to update `pip`
and `setuptools`:

```
python3 -m pip install --upgrade setuptools
python3 -m pip install --upgrade pip
```

And install the required libraries:

```
python3 -m pip install -r requirements.txt
```

The main source code is contained in `mnist.py`. To test all code at-once, run

```
python3 mnist.py
```

## Assets
- `./binaries/mnist.h5` is the trained tensorflow model.
- `./binaries/tf_mnist_cp.h5` are the model weights, which could be used to load the model from script.
- `input_schema.avsc` and `output_schema.avsc` are AVRO-compliant json files that detail the input and output schema, respectively.

## Scoring (Inference) Requests

### Sample Inputs

Choose `./data/sample_input.json` as the input file. The scoring job requires a runtime with **python >= 3.8**.

### Sample Output

`./data/sample_output.json` contains the output corrsesding to the input file above. Each line/output record is a dictionary with 2 keys: `predicted_probs` and `score`. Here's an example:
```json
{
    "predicted_probs": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.99999, 0.0, 1e-05], 
    "score": 7
}
```