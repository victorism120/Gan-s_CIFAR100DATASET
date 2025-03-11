# GAN Training on CIFAR-100 Dataset

## Overview
This project focuses on training a Generative Adversarial Network (GAN) using the CIFAR-100 dataset. The GAN consists of a Generator and a Discriminator, working in an adversarial setup to generate realistic images.

## GAN Architecture & Hyperparameters
- **Generator**: Uses transposed convolution layers with batch normalization and ReLU activation.
- **Discriminator**: Uses convolutional layers with LeakyReLU activation and dropout for stability.
- **Loss Function**: Binary cross-entropy.
- **Optimizer**: Adam optimizer with learning rate tuning.
- **Batch Size**: 64.
- **Epochs**: 100.
- **Input Shape**: 32x32x3.

## Training Observations
- **Mode Collapse**: The generator sometimes produced similar images; mitigated by tuning the learning rate and adding noise to the discriminator.
- **Vanishing Gradients**: Occurred in early epochs; resolved by using LeakyReLU in the discriminator.
- **Training Stability**: Adjusting the batch normalization layers and using feature matching helped improve stability.

## Visual Results
Comparisons between generated and real CIFAR-100 images showed that while early outputs were noisy, later epochs produced clearer images with improved texture and structure.

## Challenges & Solutions
- **Hyperparameter Sensitivity**: Finding an optimal learning rate required multiple experiments.
- **Overfitting**: Managed by using dropout in the discriminator.
- **Training Time**: GAN training is computationally intensive; using a GPU significantly reduced training time.

## How to Run the Code
1. Install required dependencies: `pip install tensorflow numpy matplotlib`.
2. Load and preprocess the CIFAR-100 dataset.
3. Train the GAN model using the provided notebook.
4. Evaluate results by visualizing generated images.

## Future Improvements
- Implementing Wasserstein GAN (WGAN) for better stability.
- Exploring advanced architectures like StyleGAN.
- Experimenting with different datasets for better generalization.

For further details, refer to the provided Jupyter Notebook (`CIFAR100DATASET.ipynb`).

