
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import numpy as np
from src.models.hrnet import get_pose_estimation_model

class VisionPipeline:
    """
    The computer vision pipeline for processing video frames.
    """
    def __init__(self, device='cpu'):
        self.device = device
        self.object_detector = fasterrcnn_resnet50_fpn(pretrained=True).to(self.device)
        self.object_detector.eval()
        self.pose_estimator = get_pose_estimation_model(pretrained=True).to(self.device)
        self.pose_estimator.eval()

    def detect_patient(self, frame):
        """
        Detects the patient in a single frame.

        Args:
            frame (torch.Tensor): The input frame.

        Returns:
            torch.Tensor: The bounding box of the detected patient.
        """
        # The model expects a list of tensors, so we add a batch dimension
        frame = frame.unsqueeze(0)
        
        with torch.no_grad():
            prediction = self.object_detector(frame)
            
        # Get the bounding box of the person with the highest score
        best_box = None
        max_score = 0
        for box, label, score in zip(prediction[0]['boxes'], prediction[0]['labels'], prediction[0]['scores']):
            # Label 1 corresponds to 'person' in the COCO dataset
            if label == 1 and score > max_score:
                max_score = score
                best_box = box
                
        return best_box

    def process_sequence(self, frames):
        """
        Processes a sequence of frames to extract pose keypoints.

        Args:
            frames (torch.Tensor): A sequence of frames.

        Returns:
            list: A list of pose keypoints for each frame.
        """
        all_keypoints = []
        for frame in frames:
            # The frame is expected to be in the format (C, H, W)
            frame = frame.permute(2, 0, 1) # HWC to CHW
            
            # Detect patient
            patient_bbox = self.detect_patient(frame)
            
            if patient_bbox is not None:
                # Crop the frame to the patient's bounding box
                x1, y1, x2, y2 = patient_bbox
                patient_crop = frame[:, int(y1):int(y2), int(x1):int(x2)]
                
                # Resize patient crop to a fixed size for the pose estimator
                patient_crop = T.Resize((256, 192))(patient_crop)
                
                # Get pose keypoints
                with torch.no_grad():
                    keypoints_heatmap = self.pose_estimator(patient_crop.unsqueeze(0))
                
                # For simplicity, we'll just take the argmax of the heatmap to get the keypoints
                keypoints = []
                for i in range(keypoints_heatmap.shape[1]):
                    heatmap = keypoints_heatmap[0, i]
                    y, x = np.unravel_index(np.argmax(heatmap.cpu().numpy()), heatmap.shape)
                    keypoints.append((x, y))

                all_keypoints.append(keypoints)
            else:
                # If no patient is detected, append None
                all_keypoints.append(None)
                
        return all_keypoints

if __name__ == '__main__':
    # Example usage
    pipeline = VisionPipeline()
    
    # Create a dummy sequence of frames
    dummy_frames = torch.from_numpy(np.random.randint(0, 255, (2, 480, 640, 3), dtype=np.uint8)).to(torch.float32)
    
    keypoints = pipeline.process_sequence(dummy_frames)
    
    print(keypoints)
