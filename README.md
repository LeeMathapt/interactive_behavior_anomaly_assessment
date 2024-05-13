# Anomaly Assessment of Interactive Behavior

Assessment of anomalous interactive behavior in animals/humans. The whole project is based on PytorchLightning.

## Model

The `model` directory prepared useful models such as (CNN-LSTM-based) autoencoder and Variational Autoencoder (VAE).

## Data

In `data` directory, the scripts are used for correspondent training paradigms, such as training in reconstruction and prediction.

The name of each script represents its function. For example, `protagonist_based_coord_trans` means coordnate transform into ego-centric coordinate that focus on interested subject.

# Training & Evaluation

Example of usage:
```
python main.py --config config/ae.yaml --phase train
```

