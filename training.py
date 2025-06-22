import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Variational Autoencoder
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)  # Mean
        self.fc22 = nn.Linear(400, 20)  # LogVar
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# Data
transform = transforms.ToTensor()
train_loader = DataLoader(
    datasets.MNIST('.', train=True, download=True, transform=transform),
    batch_size=128, shuffle=True
)

# Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):  # Short training (can be extended)
    model.train()
    train_loss = 0
    for data, _ in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print(f"Epoch {epoch}, Loss: {train_loss / len(train_loader.dataset):.4f}")

torch.save(model.state_dict(), "vae_mnist.pth")
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, losses
import matplotlib.pyplot as plt

# Load MNIST
(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.
x_train = x_train.reshape(-1, 28 * 28)

# Define encoder
latent_dim = 2

inputs = layers.Input(shape=(784,))
h = layers.Dense(512, activation='relu')(inputs)
z_mean = layers.Dense(latent_dim)(h)
z_log_var = layers.Dense(latent_dim)(h)

# Sampling layer
def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], latent_dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

z = layers.Lambda(sampling)([z_mean, z_log_var])

# Decoder
decoder_h = layers.Dense(512, activation='relu')
decoder_out = layers.Dense(784, activation='sigmoid')
h_decoded = decoder_h(z)
outputs = decoder_out(h_decoded)

# VAE model
vae = models.Model(inputs, outputs)

# Custom loss
reconstruction_loss = losses.binary_crossentropy(inputs, outputs)
reconstruction_loss *= 784
kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')

# Train
vae.fit(x_train, x_train, epochs=10, batch_size=128)

# Save encoder and decoder
encoder = models.Model(inputs, z_mean)

decoder_input = layers.Input(shape=(latent_dim,))
_decoded_h = decoder_h(decoder_input)
_decoded_out = decoder_out(_decoded_h)
decoder = models.Model(decoder_input, _decoded_out)

decoder.save("vae_decoder.h5")
