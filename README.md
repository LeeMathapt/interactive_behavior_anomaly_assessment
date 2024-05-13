# Anomaly Assessment of Interactive Behavior

Assessment of anomalous interactive behavior in animals/humans. The whole project is based on PytorchLightning.

## Model

The `model` directory list all models can used. It include:
- (CNN-LSTM-based) autoencoder
- Variational Autoencoder (VAE).

## Data

In `data` directory, the scripts are used for correspondent training paradigms. It only include two ways at the moment:
- reconstruction
- prediction

The name of each script represents its function. For example, `protagonist_based_coord_trans` means coordnate transform into ego-centric coordinate that focus on interested subject.

# Training & Evaluation

Training parameters include:
- config: details of model and datasets to use.
- phase: selectable parameter, it include `train`, `predict` and `prepare`.
- f: specifiy detail of dataset to transform.

Example of usage:
```
python main.py --config config/ae.yaml --phase train
```

