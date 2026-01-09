
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

class DeliriumDataset(Dataset):
    """
    Custom PyTorch Dataset for loading and preprocessing ICU video data.

    Args:
        data_dir (str): The root directory of the dataset.
        split (str): The dataset split, either 'train' or 'val'.
        sequence_length (int): The number of frames to include in each sequence.
        transform (callable, optional): A function/transform to apply to each frame.
    """
    def __init__(self, data_dir, split, sequence_length=30, transform=None):
        self.data_dir = os.path.join(data_dir, split)
        self.sequence_length = sequence_length
        self.transform = transform
        self.classes = ['non_delirium', 'delirium']
        self.video_files = self._get_video_files()

    def _get_video_files(self):
        video_files = []
        for label in self.classes:
            label_dir = os.path.join(self.data_dir, label)
            for video_file in os.listdir(label_dir):
                video_files.append((os.path.join(label_dir, video_file), self.classes.index(label)))
        return video_files

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        """
        Returns a single sample from the dataset.

        Returns:
            tuple: A tuple containing the sequence of frames and the corresponding label.
        """
        video_path, label = self.video_files[idx]
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while len(frames) < self.sequence_length:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert frame to tensor
            frame = torch.from_numpy(frame.copy()).to(torch.float32)

            # Apply transform if provided
            if self.transform:
                frame = self.transform(frame)
                
            frames.append(frame)
            
        cap.release()
        
        # If the video has fewer frames than sequence_length, pad with the last frame
        while len(frames) < self.sequence_length:
            frames.append(frames[-1])
            
        # Stack frames into a single tensor
        frames = torch.stack(frames)
        
        return frames, torch.tensor(label, dtype=torch.long)

if __name__ == '__main__':
    # Example usage:
    # Create dummy data directories and video files for testing
    os.makedirs('data/train/delirium', exist_ok=True)
    os.makedirs('data/train/non_delirium', exist_ok=True)
    
    # Create dummy video files
    for i in range(3):
        dummy_video_path = f'data/train/delirium/video_{i}.mp4'
        # Create a dummy video with opencv
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(dummy_video_path, fourcc, 30, (640, 480))
        for _ in range(100):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            out.write(frame)
        out.release()
        
    for i in range(5):
        dummy_video_path = f'data/train/non_delirium/video_{i}.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(dummy_video_path, fourcc, 30, (640, 480))
        for _ in range(100):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            out.write(frame)
        out.release()

    dataset = DeliriumDataset(data_dir='data', split='train', sequence_length=30)
    
    print(f"Dataset size: {len(dataset)}")
    
    frames, label = dataset[0]
    
    print(f"Frames shape: {frames.shape}")
    print(f"Label: {label}")
