
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    #code explains on docs


    def __init__(self, input_dim=784, hidden_dim=128, latent_dim=8):
        super().__init__()
# what this is doing is initializing the VAE class, which is a subclass of nn.Module. The __init__ method takes three parameters: input_dim, hidden_dim, and latent_dim, which represent the dimensions of the input data, the hidden layer, and the latent space, respectively. The default values are set to 784 for input_dim (which corresponds to 28x28 images), 128 for hidden_dim, and 8 for latent_dim.
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder_fc1 = nn.Linear(input_dim, hidden_dim)
        self.encoder_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.encoder_mu = nn.Linear(hidden_dim, latent_dim)
        self.encoder_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder_fc1 = nn.Linear(latent_dim, hidden_dim)
        self.decoder_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.decoder_out = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = F.relu(self.encoder_fc1(x))
        h = F.relu(self.encoder_fc2(h))
        mu = self.encoder_mu(h)
        logvar = self.encoder_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar, noise=None):
        std = torch.exp(0.5 * logvar)

        if noise is None:
            noise = torch.randn_like(std)

        return mu + std * noise

    def decode(self, z):
        h = F.relu(self.decoder_fc1(z))
        h = F.relu(self.decoder_fc2(h))
        return self.decoder_out(h)
# this is the forward method of the VAE class. It takes an input tensor x and an optional latent_noise tensor. The method first encodes the input x to obtain the mean (mu) and log-variance (logvar) of the latent space. Then, it reparameterizes the latent space using mu, logvar, and the optional latent_noise to obtain a latent vector z. Finally, it decodes z to produce the reconstruction logits. The method returns the reconstruction logits, mu, and logvar.
    def forward(self, x, latent_noise=None):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar, latent_noise)
        reconstruction_logits = self.decode(z)
        return reconstruction_logits, mu, logvar


if __name__ == "__main__":
    torch.manual_seed(42)

    model = VAE()
    sample = torch.rand(2, 784)

    reconstruction, mu, logvar = model(sample)

    print("Input shape:", sample.shape)
    print("Reconstruction shape:", reconstruction.shape)
    print("Latent mean shape:", mu.shape)
    print("Latent log-variance shape:", logvar.shape)
#prints and stuff