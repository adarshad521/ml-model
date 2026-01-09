
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.dataset import DeliriumDataset
from models.vision_pipeline import VisionPipeline
from models.temporal_model import TemporalModel
from utils.feature_extraction import calculate_movement_energy, calculate_postural_instability

def train(data_dir, model_path='models/temporal_model.pth', num_epochs=1, batch_size=2, learning_rate=0.001, device='cpu'):
    """
    Trains the delirium detection model.

    Args:
        data_dir (str): The root directory of the dataset.
        model_path (str): The path to save the trained model.
        num_epochs (int): The number of epochs to train for.
        batch_size (int): The batch size.
        learning_rate (float): The learning rate for the optimizer.
        device (str): The device to train on ('cpu' or 'cuda').
    """
    # Initialize dataset and dataloader
    dataset = DeliriumDataset(data_dir=data_dir, split='train', sequence_length=10)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize models
    vision_pipeline = VisionPipeline(device=device)
    temporal_model = TemporalModel(input_size=2, hidden_size=128, num_layers=2).to(device)

    # Define loss and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(temporal_model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        for i, (frames_batch, labels_batch) in enumerate(dataloader):
            # Move data to the device
            frames_batch = frames_batch.to(device)
            labels_batch = labels_batch.to(device).float()

            # Process each sequence in the batch
            features_batch = []
            for frames_sequence in frames_batch:
                # Extract keypoints
                keypoints_sequence = vision_pipeline.process_sequence(frames_sequence)

                # Extract behavioral features
                movement_energy = calculate_movement_energy(keypoints_sequence)
                postural_instability = calculate_postural_instability(keypoints_sequence)
                
                features_batch.append([movement_energy, postural_instability])

            # Convert features to a tensor
            features_tensor = torch.tensor(features_batch, dtype=torch.float32).to(device)
            
            # Add a sequence dimension
            features_tensor = features_tensor.unsqueeze(1)

            # Forward pass
            outputs = temporal_model(features_tensor)
            loss = criterion(outputs.squeeze(), labels_batch)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}')

    # Save the trained model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(temporal_model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == '__main__':
    import cv2
    import numpy as np
    # Create dummy data if it doesn't exist
    if not os.path.exists('data/train/delirium'):
        os.makedirs('data/train/delirium', exist_ok=True)
        os.makedirs('data/train/non_delirium', exist_ok=True)
        # Create dummy video files
        for i in range(1):
            dummy_video_path = f'data/train/delirium/video_{i}.mp4'
            # Create a dummy video with opencv
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(dummy_video_path, fourcc, 30, (100, 100))
            for _ in range(30):
                frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                out.write(frame)
            out.release()
            
        for i in range(1):
            dummy_video_path = f'data/train/non_delirium/video_{i}.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(dummy_video_path, fourcc, 30, (100, 100))
            for _ in range(30):
                frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                out.write(frame)
            out.release()

    train(data_dir='data')
