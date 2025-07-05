"""
Generative Adversarial Network (GAN) for 2D Heart Shape Generation.

This script implements a simple Generative Adversarial Network (GAN) from scratch
using PyTorch to learn and reproduce a 2D data distribution that forms a heart shape.

The script is divided into the following main sections:
1.  **Data Generation**: Creates a dataset of 2D points forming a heart shape
    using parametric equations. This serves as the "real" data distribution that
    the GAN will learn.
2.  **Model Definition**:
    -   **Generator**: A Multi-Layer Perceptron (MLP) that takes a 2D random noise
      vector from a latent space as input and outputs a 2D data point. Its goal
      is to generate points that are indistinguishable from the real data.
    -   **Discriminator**: An MLP that takes a 2D data point as input and
      outputs a probability score indicating whether the point is real (from the
      dataset) or fake (from the generator).
3.  **Training**:
    -   The Generator and Discriminator are trained in an adversarial manner.
    -   The training alternates between updating the Discriminator and the Generator
      on a per-epoch basis.
    -   **Discriminator Training**: It is trained to correctly classify real
      points as real (label 1) and fake points as fake (label 0).
    -   **Generator Training**: It is trained to "fool" the discriminator by
      generating points that the discriminator classifies as real (label 1).
4.  **Visualization**:
    -   During training, the script periodically saves plots of the generated
      points to a 'train_progress/' directory, allowing for visualization of
      the learning process.
    -   After training is complete, a final plot of a large number of generated
      samples is displayed to show the learned distribution.

To run the script, simply execute it. Ensure the 'train_progress/' directory exists.
"""
#%% packages
import torch
from torch.utils.data import DataLoader
from torch import nn

import math
import matplotlib.pyplot as plt
import numpy as np
from random import uniform
import os

import seaborn as sns
sns.set(rc={'figure.figsize':(12,12)})
#%% create training data
# The goal here is to create a set of 2D points that form a specific shape. 
# In this case, it's a heart. This is achieved using parametric equations,
# where the x and y coordinates are both functions of a third variable, 
# theta (θ).
TRAIN_DATA_COUNT = 1024
theta = np.array([uniform(0, 2 * np.pi) for _ in range(TRAIN_DATA_COUNT)]) # np.linspace(0, 2 * np.pi, 100)
# Generating x and y data
# The x and y coordinates are calculated using the parametric equations for a heart shape.
# The equations are derived from the polar coordinates of a heart shape.
# The x coordinate is scaled by 16 and the y coordinate is a combination of cosine functions
# that create the heart shape.
x = 16 * ( np.sin(theta) ** 3 )
# The y coordinate is a combination of cosine functions that create the heart shape.
# The coefficients and the number of cosine terms are chosen to create the desired shape.
# The y coordinate is scaled by 13 and adjusted with several cosine terms to create the heart
y = 13 * np.cos(theta) - 5 * np.cos(2*theta) - 2 * np.cos(3*theta) - np.cos(4*theta)
# Finally, this line uses the seaborn library to create a scatter plot of the generated
# (x, y) points. This visualizes the dataset, showing the heart shape that the GAN 
# will be trained to reproduce.
sns.scatterplot(x=x, y=y)

#%% prepare tensors and dataloader
# Convert the x and y data into a PyTorch tensor.
# The x and y coordinates are stacked together to form a 2D tensor, where each
# row represents a point in the 2D space.
train_data = torch.Tensor(np.stack((x, y), axis=1)) # shape (1024, 2)

# Create labels for the training data. In this case, all labels are set to 0,
# indicating that these are real samples. The labels are not used in the GAN training,
# but they are included to match the expected input format for the discriminator.
# The labels are created as a tensor of zeros with the same length as the number of training
# samples.
train_labels = torch.zeros(TRAIN_DATA_COUNT)
# The training data is then combined into a list of tuples, where each tuple contains
# a data point and its corresponding label. This is done to create a dataset that can be
# easily iterated over during training. Each tuple consists of a 2D point (x, y).
train_set = [
    (train_data[i], train_labels[i]) for i in range(TRAIN_DATA_COUNT)
]

#  dataloader
BATCH_SIZE = 64
train_loader = DataLoader(
    train_set, batch_size=BATCH_SIZE, shuffle=True
)
#%% initialize discriminator and generator
# The discriminator is a neural network that takes a 2D point as input and outputs a probability
# indicating whether the point is real (from the training data) or generated (from the generator).
# The generator is a neural network that takes a random noise vector as input and outputs a
# 2D point that is intended to resemble the training data.
discriminator = nn.Sequential(
            nn.Linear(2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

# The generator is a neural network that takes a 2D noise vector as input and outputs a
# 2D point that is intended to resemble the training data. The generator is designed to
# learn to produce points that the discriminator will classify as real.
generator = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            # The output layer has 2 neurons, one for each dimension of the 2D point.
            # The output is not activated by a sigmoid function because the generator
            # is expected to produce points in the same range as the training data.
            # The generator will learn to produce points that resemble the training data.
            nn.Linear(32, 2),
        )

# %% training
LR = 0.001
NUM_EPOCHS = 3000
loss_function = nn.BCELoss()
optimizer_discriminator = torch.optim.Adam(discriminator.parameters())
optimizer_generator = torch.optim.Adam(generator.parameters())

# Create directory for saving training progress images if it doesn't exist
os.makedirs("train_progress", exist_ok=True)

# Outer loop for epochs
# In each epoch, the discriminator is trained on both real samples from the training data and 
# generated samples from the generator.
# The generator is trained to produce samples that the discriminator classifies as real.
# The training alternates between updating the discriminator and the generator, with the 
# discriminator being updated every two epochs and the generator being updated every other epoch.
for epoch in range(NUM_EPOCHS):
    # Inner loop for batches
    for n, (real_samples, _) in enumerate(train_loader):
        # Data for training the discriminator
        real_samples_labels = torch.ones((BATCH_SIZE, 1)) # Labels for real samples is 1
        # Creating random noise vectors for the generator.
        # These vectors are sampled from a normal distribution and will be used as input to the generator
        # to generate new samples. The noise vectors are of size 2, matching the output
        # dimension of the generator.
        latent_space_samples = torch.randn((BATCH_SIZE, 2))
        # The generator generates samples from the latent space.
        # The latent space samples are random noise vectors that the generator will transform
        # into 2D points. The generator learns to produce points that resemble the training data
        # by adjusting its weights based on the feedback from the discriminator.
        generated_samples = generator(latent_space_samples)
        generated_samples_labels = torch.zeros((BATCH_SIZE, 1)) # Labels for generated samples is 0
        all_samples = torch.cat((real_samples, generated_samples))
        all_samples_labels = torch.cat(
            (real_samples_labels, generated_samples_labels)
        )

        if epoch % 2 == 0:
            # Training the discriminator
            discriminator.zero_grad()
            # Forward pass through the discriminator
            # The discriminator processes both real and generated samples.
            # It takes the concatenated samples and their corresponding labels as input.
            output_discriminator = discriminator(all_samples)
            # The discriminator's output is compared to the labels of real and generated samples.
            # The loss function measures how well the discriminator can distinguish between real 
            # and generated samples.
            loss_discriminator = loss_function(
                output_discriminator, all_samples_labels)
            loss_discriminator.backward()
            optimizer_discriminator.step()

        if epoch % 2 == 1:
            # Data for training the generator
            latent_space_samples = torch.randn((BATCH_SIZE, 2))
            # Training the generator
            generator.zero_grad()
            generated_samples = generator(latent_space_samples)
            output_discriminator_generated = discriminator(generated_samples)
            loss_generator = loss_function(
                output_discriminator_generated, real_samples_labels
            )
            loss_generator.backward()
            optimizer_generator.step()

    # Show progress at the end of the epoch, not in the batch loop
    if epoch % 100 == 1:
        # This condition implies the epoch is odd, so only generator was trained in this epoch.
        # The `loss_generator` variable holds the value from the last batch.
        print(f"\n--- Epoch {epoch} ---")
        print(f"Last batch Generator Loss: {loss_generator.item():.4f}")
        with torch.no_grad():
            latent_space_samples = torch.randn(1000, 2)
            generated_samples = generator(latent_space_samples).detach()
        plt.figure()
        plt.plot(generated_samples[:, 0], generated_samples[:, 1], ".")
        plt.xlim((-20, 20))
        plt.ylim((-20, 15))
        plt.text(10, 15, f"Epoch {epoch}")
        plt.savefig(f"train_progress/image{str(epoch).zfill(3)}.jpg")
        plt.show()
        plt.close()  # Close the figure after displaying to free up memory


    

# %% check the result
# Generate a random set of points from the latent space and visualize the generated samples.
# This is done after the training loop to see how well the generator has learned to produce samples
# that resemble the heart shape.
latent_space_samples = torch.randn(10000, 2)
# The generator is used to transform the random noise vectors into 2D points.
# These points are expected to form a distribution that resembles the heart shape learned during training.
generated_samples = generator(latent_space_samples)
# Detach the generated samples from the computation graph to avoid memory issues during plotting.
generated_samples = generated_samples.detach()
plt.plot(generated_samples[:, 0], generated_samples[:, 1], ".")
plt.text(10, 15, f"Epoch {epoch}")
plt.show()

# %%