
import numpy as np

def calculate_movement_energy(keypoints_sequence):
    """
    Calculates the movement energy of a sequence of keypoints.

    Args:
        keypoints_sequence (list): A list of keypoints for each frame in a sequence.
                                  Each element is a list of (x, y) tuples.

    Returns:
        float: The total movement energy.
    """
    total_energy = 0
    for i in range(1, len(keypoints_sequence)):
        prev_keypoints = keypoints_sequence[i-1]
        curr_keypoints = keypoints_sequence[i]
        
        if prev_keypoints is not None and curr_keypoints is not None:
            prev_keypoints = np.array(prev_keypoints)
            curr_keypoints = np.array(curr_keypoints)
            
            # Ensure the keypoints arrays have the same shape
            if prev_keypoints.shape == curr_keypoints.shape:
                energy = np.sum((curr_keypoints - prev_keypoints)**2)
                total_energy += energy
                
    return total_energy

def calculate_postural_instability(keypoints_sequence, threshold=1000):
    """
    Calculates the postural instability of a sequence of keypoints.

    Args:
        keypoints_sequence (list): A list of keypoints for each frame in a sequence.
        threshold (int): The threshold for detecting a significant posture shift.

    Returns:
        int: The number of posture shifts.
    """
    posture_shifts = 0
    for i in range(1, len(keypoints_sequence)):
        prev_keypoints = keypoints_sequence[i-1]
        curr_keypoints = keypoints_sequence[i]
        
        if prev_keypoints is not None and curr_keypoints is not None:
            prev_keypoints = np.array(prev_keypoints)
            curr_keypoints = np.array(curr_keypoints)

            if prev_keypoints.shape == curr_keypoints.shape:
                distance = np.sum((curr_keypoints - prev_keypoints)**2)
                if distance > threshold:
                    posture_shifts += 1
                    
    return posture_shifts

if __name__ == '__main__':
    # Example usage
    # Create a dummy sequence of keypoints
    dummy_keypoints_sequence = []
    for _ in range(30):
        frame_keypoints = []
        for _ in range(17):
            frame_keypoints.append((np.random.randint(0, 640), np.random.randint(0, 480)))
        dummy_keypoints_sequence.append(frame_keypoints)

    # Introduce a significant shift
    dummy_keypoints_sequence[15] = [(x + 100, y + 100) for x, y in dummy_keypoints_sequence[15]]
    
    movement_energy = calculate_movement_energy(dummy_keypoints_sequence)
    postural_instability = calculate_postural_instability(dummy_keypoints_sequence)
    
    print(f"Movement Energy: {movement_energy}")
    print(f"Postural Instability: {postural_instability}")
