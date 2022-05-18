# GANs
Hello, this repo is a collection of different Generative adversarial networks; it is done for self-training and the proper recreation of the paper architectures because many implementations contain inaccuracies. Also, I've added the `pytorch-lightning` package as a core for the development because I believe the train loop here will be quite the same.

## Environment
You can use any environment manager for your system, here is the example
according to `conda`:
```
conda create -n gans python=3.9
conda activate gans
pip install -r requirements.txt
```

## Config
Here we have an example for the default config to specify the use of the system:
```
seed: 4567
batch_size: 32
gan_type: "gan"
latent_dim: 100
train:
  epochs: 40
  lr: 0.0002
  b1: 0.5
  b2: 0.999
```

## Data
I have used [BIRDS 400 - SPECIES IMAGE CLASSIFICATION](https://www.kaggle.com/datasets/gpiosenka/100-bird-species) dataset. Because it is not a trivial dataset for the generation, it is not as giant as it can be.

## GANs list:
1. GAN[1]
2. DCGAN[2]
3. BEGAN[3]
4. WGAN[4]

## References
1. [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)
2. [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)
3. [BEGAN: Boundary Equilibrium Generative Adversarial Networks](https://arxiv.org/abs/1703.10717)
4. [Wasserstein GAN](https://arxiv.org/abs/1701.07875)
