
import argparse
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from train import train

def main():
    parser = argparse.ArgumentParser(description='ICU Delirium Detection')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'inference'],
                        help='The mode to run the application in.')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='The root directory of the dataset.')
    parser.add_argument('--model_path', type=str, default='models/temporal_model.pth',
                        help='The path to the trained model.')
    parser.add_argument('--video_path', type=str, default='data/train/delirium/video_0.mp4',
                        help='The path to the input video for inference.')
    parser.add_argument('--num_epochs', type=int, default=1,
                        help='The number of epochs to train for.')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='The batch size.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='The learning rate for the optimizer.')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help='The device to train on.')
    
    args = parser.parse_args()

    if args.mode == 'train':
        # Create dummy data if it doesn't exist
        if not os.path.exists(os.path.join(args.data_dir, 'train/delirium')):
            import cv2
            import numpy as np
            os.makedirs(os.path.join(args.data_dir, 'train/delirium'), exist_ok=True)
            os.makedirs(os.path.join(args.data_dir, 'train/non_delirium'), exist_ok=True)
            # Create dummy video files
            for i in range(1):
                dummy_video_path = os.path.join(args.data_dir, f'train/delirium/video_{i}.mp4')
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(dummy_video_path, fourcc, 30, (100, 100))
                for _ in range(30):
                    frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                    out.write(frame)
                out.release()
                
            for i in range(1):
                dummy_video_path = os.path.join(args.data_dir, f'train/non_delirium/video_{i}.mp4')
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(dummy_video_path, fourcc, 30, (100, 100))
                for _ in range(30):
                    frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                    out.write(frame)
                out.release()

        train(data_dir=args.data_dir, model_path=args.model_path, num_epochs=args.num_epochs, 
              batch_size=args.batch_size, learning_rate=args.learning_rate, device=args.device)
    elif args.mode == 'inference':
        inference(video_path=args.video_path, model_path=args.model_path, device=args.device)

def inference(video_path, model_path, device='cpu'):
    import torch
    import cv2
    import numpy as np
    from src.models.vision_pipeline import VisionPipeline
    from src.models.temporal_model import TemporalModel
    from src.utils.feature_extraction import calculate_movement_energy, calculate_postural_instability

    # Initialize models
    vision_pipeline = VisionPipeline(device=device)
    temporal_model = TemporalModel(input_size=2, hidden_size=128, num_layers=2).to(device)
    
    # Load the trained model weights
    temporal_model.load_state_dict(torch.load(model_path))
    temporal_model.eval()

    # Open video file
    cap = cv2.VideoCapture(video_path)
    frames = []
    sequence_length = 10

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to tensor
        frame = torch.from_numpy(frame.copy()).to(torch.float32)
        frames.append(frame)

        if len(frames) == sequence_length:
            # Process the sequence
            frames_tensor = torch.stack(frames)
            keypoints_sequence = vision_pipeline.process_sequence(frames_tensor)

            # Extract behavioral features
            movement_energy = calculate_movement_energy(keypoints_sequence)
            postural_instability = calculate_postural_instability(keypoints_sequence)
            
            features = [movement_energy, postural_instability]

            # Convert features to a tensor
            features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
            features_tensor = features_tensor.unsqueeze(0).unsqueeze(1) # Add batch and sequence dimensions

            # Predict risk score
            with torch.no_grad():
                risk_score = temporal_model(features_tensor)
            
            print(f"Delirium Risk Score: {risk_score.item():.4f}")

            # Reset frames for the next sequence
            frames = []

    cap.release()

if __name__ == '__main__':
    main()
