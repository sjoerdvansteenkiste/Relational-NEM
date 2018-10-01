# Relational Neural Expectation-Maximization

![r-nem](animations/balls4mass_rnem1.gif)&nbsp;
![r-nem](animations/balls4mass_rnem2.gif)&nbsp;
![r-nem](animations/balls4mass_rnem3.gif)&nbsp;
![r-nem](animations/balls4mass_rnem4.gif)&nbsp;&nbsp;&nbsp;
![r-nem](animations/balls678mass_rnem1.gif)&nbsp;
![r-nem](animations/balls678mass_rnem2.gif)&nbsp;
![r-nem](animations/balls678mass_rnem3.gif)&nbsp;
![r-nem](animations/balls678mass_rnem4.gif)&nbsp;&nbsp;&nbsp;
![r-nem](animations/curtain_rnem1.gif)&nbsp;
![r-nem](animations/curtain_rnem2.gif)&nbsp;
![r-nem](animations/curtain_rnem3.gif)&nbsp;
![r-nem](animations/curtain_rnem4.gif)

This is the code repository complementing the paper ["Relational Neural Expectation Maximization: Unsupervised Discovery of Objects and their Interactions"](https://openreview.net/pdf?id=ryH20GbRW). All experiments from the paper
can be reproduced from this repository. Data and pre-trained models are available [here](https://www.dropbox.com/sh/8fjscaromuu094o/AAB4Ye7yD3lmUd-iInbEfcHga?dl=0).

## Dependencies and Setup

- tensorflow==1.2.1
- numpy >= 1.14.0
- sacred == 0.7.2
- pymongo == 3.6.0
- Pillow == 5.0.0
- scipy >= 1.0.0
- scikit-learn >= 0.19.1
- scikit-image >= 0.13.1
- matplotlib >= 2.1.2
- h5py >= 2.7.1

## Experiments

### Training
Use the following calls to train R-NEM (and baselines) for each experiment. Data is provided for up to 50 timesteps.

#### Bouncing Balls with Mass / Occluding Curtain

The configurations below train by default on the bouncing balls dataset with variable mass. Use `dataset.balls3curtain64` in stead of `dataset.balls4mass64` to train on the bouncing balls dataset with the occluding curtain.

R-NEM

```bash
python nem.py with dataset.balls4mass64 network.r_nem nem.k=5
```

R-NEM (K=8)

```bash
python nem.py with dataset.balls4mass64 network.r_nem nem.k=8
```

R-NEM (no attention)

```bash
python nem.py with dataset.balls4mass64 network.r_nem_no_attention nem.k=5
```

RNN-EM

```bash
python nem.py with dataset.balls4mass64 network.rnn_250 nem.k=5
```

RNN

```bash
python nem.py with dataset.balls4mass64 network.rnn_250 nem.k=1
```

LSTM

```bash
python nem.py with dataset.balls4mass64 network.lstm_250 nem.k=1
```


#### Atari

R-NEM

```bash
python nem.py with dataset.atari network.enc_dec_84_atari network.r_nem_actions no_score no_collisions nem.k=4 nem.nr_steps=25 training.batch_size=32 feed_actions=True noise.prob=0.002
```

### Evaluation

In order to evaluate a trained model on the test set (potentially with a different number of components) use the `run_from_file` command. For example, having trained R-NEM on balls4mass64 using the config above, one could evaluate it on the test set with 6-8 balls by calling:

```bash
python nem.py run_from_file with dataset.balls678mass64 network.r_nem nem.k=8
```

Note that by default the network path is set to the log_dir (debug_out), but can alternatively be set with `net_path`. By default the best model is used. Pre-trained models are available [here](https://www.dropbox.com/sh/8fjscaromuu094o/AAB4Ye7yD3lmUd-iInbEfcHga?dl=0).

#### Rollout

In order to simulate the environment for a number of timesteps use the `rollout_from_file` command. The number of simulation steps can be controlled with `run_config.rollout_steps`, which occur after taking `nem.nr_steps` of normal steps.
