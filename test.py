import torch
from models.generator import Generator

def run_test():
    # Setup parameters
    latent_dim = 100
    output_dim = 784 # Example size for a small image/packet

    # Initialize Generator
    gen = Generator(latent_dim, output_dim)
    
    # Create random noise (Latent Vector)
    noise = torch.randn(1, latent_dim)
    
    # Generate "Fake" data
    fake_data = gen(noise)
    
    print(f"Test Successful!")
    print(f"Generated data shape: {fake_data.shape}")

if __name__ == "__main__":
    run_test()