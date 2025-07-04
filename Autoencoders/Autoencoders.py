#%% packages
# Let's start by importing all the tools we'll need for our project.
from typing import OrderedDict # This is used for type hinting, making our code easier to read.
import torch # This is the main PyTorch library, our primary tool for building neural networks.
import torch.nn as nn # A module from PyTorch that provides building blocks for our models, like layers.
import torch.nn.functional as F # This gives us access to helpful functions like our loss function.
import torchvision.transforms as transforms # A tool for applying common image transformations.
from torch.utils.data import DataLoader # Helps us load our data in manageable batches.
from torchvision.datasets import ImageFolder # A handy class for loading image data from a folder.
import numpy as np # A popular library for numerical operations, especially for handling arrays.
import matplotlib.pyplot as plt # Our go-to library for plotting and visualizing images.
import torchvision.utils # Provides helpful utilities for working with image tensors, like creating a grid.

#%% Dataset and data loader
# Here, we'll set up our dataset and prepare it for training.

# This is the path to the directory where our training images are stored.
path_images = 'data/train'

# Image transformations
# We define a sequence of transformations to apply to each image as it's loaded.
# This ensures all our images are consistent in size, color, and format.
transform = transforms.Compose(
    [transforms.Resize((64,64)), # First, we resize every image to be 64x64 pixels.
    transforms.Grayscale(num_output_channels=1), # Then, we convert the images to grayscale (1 color channel).
    transforms.ToTensor(), # Next, we convert the image data into a PyTorch Tensor.
    transforms.Normalize((0.5, ), (0.5, ))]) # Finally, we normalize the pixel values to be in the range [-1, 1].

# Load dataset
# The ImageFolder class is great because it automatically finds images in a folder
# and assumes subdirectories are different classes (though we don't use the labels here).
# ImageFolder expects a directory structure like:
# data/train/class1/xxx.png
# data/train/class2/yyy.png
dataset = ImageFolder(root=path_images, transform=transform) # We apply our transformations to each image.

# Create DataLoader
# The DataLoader takes our dataset and prepares it to be fed to the model in small groups, or "batches".
dataloader = DataLoader(
    dataset,      # The dataset we want to load.
    batch_size=8, # We'll process 8 images at a time.
    shuffle=True  # Shuffling the data at each epoch helps the model learn better and not get stuck.
)

# %% model class
# Now, let's define the architecture of our autoencoder.

# This constant defines the size of our "bottleneck" or latent space.
# It's the compressed representation of the image.
LATENT_DIMS = 128

class Encoder(nn.Module):
    """
    The Encoder's job is to take an image and compress it down into a smaller,
    dense representation called the "latent vector". Think of it as creating a
    summary of the image's most important features.

    It does this using a series of convolutional layers to detect features,
    and then a fully connected layer to produce the final latent vector.
    """
    def __init__(self) -> None:
        """
        Here we define all the layers that make up the Encoder.
        """
        super().__init__() # This is a standard and necessary call to the parent class constructor.
        
        # Convolutional layers
        # This layer finds basic patterns in the 1-channel (grayscale) input image.
        self.conv1 = nn.Conv2d(1, 6, 3)  # Input channels: 1, Output channels: 6, Kernel size: 3x3
        
        # This layer finds more complex patterns in the features from the first layer.
        self.conv2 = nn.Conv2d(6, 16, 3) # Input channels: 6, Output channels: 16, Kernel size: 3x3
        
        # The ReLU activation function introduces non-linearity, allowing the model to learn more complex patterns.
        self.relu = nn.ReLU()
        
        # This layer flattens the 2D feature maps into a single long 1D vector.
        self.flatten = nn.Flatten() # This prepares the data for the fully connected layer.
        
        # This fully connected layer takes the flattened vector and compresses it down to our latent dimension size.
        self.fc = nn.Linear(16*60*60, LATENT_DIMS) # Input size: 16*60*60, Output size: 128

    def forward(self, x):
        """
        This function defines the "forward pass" – how data flows through the layers.
        
        Args:
            x (torch.Tensor): The input image tensor.

        Returns:
            torch.Tensor: The compressed latent vector.
        """
        # Pass the input through the first convolutional layer, then apply ReLU.
        x = self.relu(self.conv1(x))
        # Pass the result through the second convolutional layer, then apply ReLU.
        x = self.relu(self.conv2(x))
        # Flatten the feature maps into a vector.
        x = self.flatten(x)
        # Pass the vector through the fully connected layer to get the latent representation.
        x = self.fc(x)
        # Return the final compressed vector.
        return x

class Decoder(nn.Module):
    '''
    The Decoder's job is the opposite of the Encoder's. It takes the compressed
    latent vector and tries to reconstruct the original image from it.

    It uses a fully connected layer to expand the vector, and then "transposed
    convolutional" layers to upsample the features back into an image.
    '''
    def __init__(self) -> None:
        """
        Here we define the layers that make up the Decoder.
        """
        super().__init__() # Standard call to the parent class constructor.
        
        # This fully connected layer takes the latent vector and expands it back to the size it was before flattening in the encoder.
        self.fc = nn.Linear(LATENT_DIMS, 16*60*60) # Input size: 128, Output size: 16*60*60
        
        # Transposed convolutions are like "un-doing" a convolution, they upsample the data.
        self.conv2 = nn.ConvTranspose2d(16, 6, 3) # This layer upsamples from 16 channels to 6.
        self.conv1 = nn.ConvTranspose2d(6, 1, 3)  # This layer upsamples from 6 channels to our final 1 channel (grayscale image).
        
        # We'll use the same ReLU activation function.
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
        This function defines how the latent vector is transformed back into an image.

        Args:
            x (torch.Tensor): The latent vector from the encoder.

        Returns:
            torch.Tensor: The reconstructed image tensor.
        """
        # First, pass the latent vector through the fully connected layer.
        x = self.fc(x)
        # Reshape the 1D vector back into 2D feature maps for the convolutional layers.
        # The '-1' tells PyTorch to automatically calculate the batch size.
        x = x.view(-1, 16, 60, 60)
        # Pass through the first transposed convolution, then apply ReLU.
        x = self.relu(self.conv2(x))
        # Pass through the final transposed convolution to get the image, then apply ReLU.
        x = self.relu(self.conv1(x))
        # Return the reconstructed image.
        return x

class Autoencoder(nn.Module):
    """
    The Autoencoder class brings the Encoder and Decoder together into a single model.
    It takes an image, encodes it, and then decodes it. The goal is for the output
    image to be as close to the input image as possible.
    """
    def __init__(self) -> None:
        """
        Initializes the Autoencoder by creating an instance of the Encoder and Decoder.
        """
        super().__init__() # Standard call to the parent class constructor.
        self.encoder = Encoder() # Create our encoder.
        self.decoder = Decoder() # Create our decoder.
    
    def forward(self, x):
        """
        Defines the full forward pass: from input image to reconstructed image.

        Args:
            x (torch.Tensor): The input image tensor.

        Returns:
            torch.Tensor: The reconstructed image tensor.
        """
        # Step 1: Pass the input image through the encoder to get the latent vector.
        x = self.encoder(x)
        # Step 2: Pass the latent vector through the decoder to get the reconstructed image.
        x = self.decoder(x)
        # Return the final output.
        return x

# Test it
# This is a quick sanity check to make sure our model works as expected.
input = torch.rand((1, 1, 64, 64)) # Create a random tensor with the same shape as one of our images.
model = Autoencoder() # Create an instance of our autoencoder.
model(input).shape # Pass the random tensor through the model and check the output shape.


#%% init model, loss function, optimizer
# Now we'll set up everything we need to train our model.

model = Autoencoder() # Create a new instance of our model for training.

# The optimizer is what updates the model's weights to reduce the loss.
# Adam is a popular and effective choice. We tell it which parameters to optimize (all of them)
# and set a learning rate (how big of a step to take).
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# This is the number of times we will loop through the entire dataset.
NUM_EPOCHS = 30

# Let's start the training loop!
for epoch in range(NUM_EPOCHS):
    # We'll keep track of the loss for each batch in this epoch.
    losses_epoch = [] 
    # This inner loop iterates over the DataLoader, one batch at a time.
    for batch_idx, (data, target) in enumerate(dataloader):
        # Make sure the data has the correct shape for our model.
        data = data.view(-1, 1, 64, 64) 
        # Get the model's output (the reconstructed image) for the current batch.
        output = model(data) 

        # Calculate the loss. Mean Squared Error (MSE) measures the average squared
        # difference between the pixels of the original image and the reconstructed one.
        # Our goal is to make this value as small as possible.
        loss = F.mse_loss(output, data) 
        losses_epoch.append(loss.item()) # Store the loss for this batch.

        # These three steps are the core of training in PyTorch:
        optimizer.zero_grad() # 1. Reset the gradients from the previous step.
        loss.backward()       # 2. Calculate the gradients for this step (backpropagation).
        optimizer.step()      # 3. Update the model's weights using the optimizer.
    
    # At the end of each epoch, print the average loss to see how our training is going.
    print(f"Epoch: {epoch} \tLoss: {np.mean(losses_epoch)}")

# %% visualise original and reconstructed images
# Now that the model is trained, let's see how well it did!

def show_image(img):
    """
    A helper function to display an image tensor. It first denormalizes the image
    and then converts it to the right format for plotting with Matplotlib.

    Args:
        img (torch.Tensor): The image tensor to display.
    """
    # Remember we normalized the images to [-1, 1]? This reverses that process.
    img = 0.5 * (img + 1)
    # Convert the PyTorch tensor to a NumPy array, which Matplotlib can use.
    npimg = img.numpy() 
    # Matplotlib expects the color channels to be the last dimension, but PyTorch
    # has them first. np.transpose rearranges the dimensions.
    plt.imshow(np.transpose(npimg, (1, 2, 0))) 

# Get one batch of images from our dataloader to visualize.
images, labels = next(iter(dataloader))

# --- Show the original images ---
print('Original Images')
plt.rcParams["figure.figsize"] = (20,3) # Make the plot wider to see all images clearly.
# `make_grid` arranges the batch of images into a single grid for easy viewing.
show_image(torchvision.utils.make_grid(images))
plt.show() # Display the plot.

# %% latent space
# Let's visualize what the compressed "latent space" representation looks like.
print('Latent Space Representation (Visualized as an Image)')
# Pass the original images through the encoder only.
latent_img = model.encoder(images)
# The latent vector is 1D (size 128). To visualize it, we can reshape it into a small 2D "image".
latent_img = latent_img.view(-1, 1, 8, 16) # Reshape to a batch of 8x16 single-channel images.
# Use our helper function to show this "latent image".
show_image(torchvision.utils.make_grid(latent_img.detach())) # We use .detach() because we don't need gradients here.
plt.show() # Display the plot.

#%%
print('Reconstructed Images')
# Pass the original images through the full autoencoder to get the reconstructions.
# We use .detach() again to remove the tensors from the computation graph.
reconstructed_images = model(images).detach()
# Show the grid of reconstructed images.
show_image(torchvision.utils.make_grid(reconstructed_images))
plt.show() # Display the plot.

# %% Compression rate
# Finally, let's calculate how much we're compressing the data.

# The size of the original image is its height * width * channels.
image_size = images.shape[2] * images.shape[3] * 1 # 64 * 64 * 1 = 4096

# Compression rate shows how much smaller the latent representation is compared to the original image.
# Formula: (1 - (compressed_size / original_size)) * 100
compression_rate = (1 - LATENT_DIMS / image_size) * 100

# Print the result!
print(f"Original image size: {image_size} values")
print(f"Latent space size: {LATENT_DIMS} values")
print(f"Compression rate: {compression_rate:.2f}%")
