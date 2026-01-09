
import torch
import torch.nn as nn

class TemporalModel(nn.Module):
    """
    An LSTM-based model for delirium risk prediction.
    """
    def __init__(self, input_size, hidden_size=128, num_layers=2):
        super(TemporalModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): A tensor of shape (batch_size, sequence_length, input_size).

        Returns:
            torch.Tensor: A tensor of shape (batch_size, 1) containing the delirium risk score.
        """
        # Get the LSTM output
        lstm_out, _ = self.lstm(x)
        
        # Take the output from the last time step
        last_output = lstm_out[:, -1, :]
        
        # Pass through the fully connected layer and sigmoid
        risk_score = self.sigmoid(self.fc(last_output))
        
        return risk_score

if __name__ == '__main__':
    # Example usage
    # Create a dummy sequence of behavioral features
    # batch_size = 4, sequence_length = 30, input_size = 2 (movement_energy, postural_instability)
    dummy_features = torch.randn(4, 30, 2)
    
    # Initialize the model
    model = TemporalModel(input_size=2)
    
    # Get the risk score
    risk_score = model(dummy_features)
    
    print(f"Risk score shape: {risk_score.shape}")
    print(f"Risk scores: {risk_score}")
