# Supported Algorithms
The FedMedICL framework supports a variety of algorithms. Depending on the algorithm specified, different settings will be applied automatically. 

The `--algorithm` parameter supports the following baselines:
- `ERM`: Empirical Risk Minimization, standard training without federated learning.
- [`fedavg`](https://arxiv.org/abs/1602.05629): Federated Averaging, aggregates model updates from multiple clients.
- [`mixup`](https://arxiv.org/abs/1710.09412): F-MixUp, applies mixup data augmentation in a federated setting.
- [`SWAD`](https://arxiv.org/abs/2102.08604): F-SWAD, applies stochastic weight averaging in a federated setting.
- [`ER`](https://arxiv.org/abs/1902.10486): F-ER, uses experience replay in a federated learning context.
- `resampling`: F-GB, applies group balanced resampling in federated learning.
- [`CRT`](https://arxiv.org/abs/1910.09217): F-CRT, employs class rebalanced training in a federated learning setup.
- `CB`: F-CB, utilizes class balanced sampling in federated learning.