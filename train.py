import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# --- 1. Defense-Grade GAN Architecture ---

class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, output_dim),
            nn.Tanh() # Normalizes signal between -1 and 1
        )

    def forward(self, x):
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid() # Probability: Real vs Spoofed
        )

    def forward(self, x):
        return self.main(x)

# --- 2. Training Setup ---

# Hyperparameters
latent_dim = 100
signal_dim = 64  # Length of the communication burst
lr = 0.0002
batch_size = 64
epochs = 5000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gen = Generator(latent_dim, signal_dim).to(device)
disc = Discriminator(signal_dim).to(device)

criterion = nn.BCELoss()
opt_gen = optim.Adam(gen.parameters(), lr=lr)
opt_disc = optim.Adam(disc.parameters(), lr=lr)

# --- 3. Training Loop ---

print(f"Starting WhisperNet Training on {device}...")

for epoch in range(epochs):
    # Create "Real" Defense Signals (Dummy data for demo)
    real_signals = torch.randn(batch_size, signal_dim).to(device)
    
    # Labels
    label_real = torch.ones(batch_size, 1).to(device)
    label_fake = torch.zeros(batch_size, 1).to(device)

    # --- Train Discriminator: Maximize log(D(x)) + log(1 - D(G(z))) ---
    noise = torch.randn(batch_size, latent_dim).to(device)
    fake_signals = gen(noise)
    
    disc_real = disc(real_signals)
    loss_D_real = criterion(disc_real, label_real)
    
    disc_fake = disc(fake_signals.detach())
    loss_D_fake = criterion(disc_fake, label_fake)
    
    loss_D = (loss_D_real + loss_D_fake) / 2
    
    opt_disc.zero_grad()
    loss_D.backward()
    opt_disc.step()

    # --- Train Generator: Maximize log(D(G(z))) ---
    output = disc(fake_signals)
    loss_G = criterion(output, label_real)

    opt_gen.zero_grad()
    loss_G.backward()
    opt_gen.step()

    if epoch % 500 == 0:
        print(f"Epoch [{epoch}/{epochs}] | Loss D: {loss_D:.4f}, Loss G: {loss_G:.4f}")

# Save the models for deployment
torch.save(gen.state_state_dict(), "whispernet_gen.pth")
print("Model saved as whispernet_gen.pth")