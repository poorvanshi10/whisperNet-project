🎭 WhisperNet: Generative Adversarial Network for Secure Data

**WhisperNet** is an AI-driven project that utilizes **Generative Adversarial Networks (GANs)** to synthesize secure communication patterns. This project was developed as a submission for the Visionx 2.O Ideathon by Startup India, where it secured **1st Place**.



🚀 Overview
WhisperNet consists of two neural networks—the **Generator** and the **Discriminator**—competing in a zero-sum game to produce high-fidelity synthetic data.

- **Generator:** Learns to create data that mimics secure communication packets.
- **Discriminator:** Learns to distinguish between real secure data and the "fake" data created by the generator.

📁 Project Structure
```text
WhisperNet-project/
├── models/
│   ├── generator.py      # Upgraded architecture with BatchNorm
│   └── discriminator.py  # Robust architecture with Dropout
├── data/                 # (Future) Training datasets
├── train.py              # Main training logic
├── test.py               # Quick validation script
└── requirements.txt      # Dependency list