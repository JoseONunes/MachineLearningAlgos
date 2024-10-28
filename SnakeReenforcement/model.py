import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

# Set device for GPU compatibility
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        # Reduced learning rate for stability
        self.lr = 0.0001
        self.gamma = gamma
        self.model = model.to(device)  # Ensure model is on the correct device
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        try:
            # Move tensors to the device
            state = torch.tensor(state, dtype=torch.float).to(device)
            next_state = torch.tensor(next_state, dtype=torch.float).to(device)
            action = torch.tensor(action, dtype=torch.long).to(device)
            reward = torch.tensor(reward, dtype=torch.float).to(device)
            
            # Ensure tensors have a batch dimension
            if state.ndim == 1:
                state = state.unsqueeze(0)
            if next_state.ndim == 1:
                next_state = next_state.unsqueeze(0)
            if action.ndim == 1:
                action = action.unsqueeze(0)
            if reward.ndim == 0:  # If reward is a scalar, add batch and feature dimensions
                reward = reward.unsqueeze(0).unsqueeze(0)
            elif reward.ndim == 1:
                reward = reward.unsqueeze(1)

            # Debugging checks for tensor shapes and data types
            assert state.ndim == 2, f"State shape is unexpected: {state.shape}"
            assert next_state.ndim == 2, f"Next state shape is unexpected: {next_state.shape}"
            assert action.ndim == 2, f"Action shape is unexpected: {action.shape}"
            assert reward.ndim == 2, f"Reward shape is unexpected: {reward.shape}"
            assert isinstance(done, bool), "Expected 'done' to be a boolean"
            
            print(f"State shape: {state.shape}, Next state shape: {next_state.shape}")
            print(f"Action shape: {action.shape}, Reward shape: {reward.shape}")

            # Predicted Q values with current state, using no_grad to save memory
            with torch.no_grad():
                try:
                    pred = self.model(state)
                    # Check for NaN or Inf in predictions
                    if torch.isnan(pred).any() or torch.isinf(pred).any():
                        print("Detected NaN or Inf in predictions.")
                        return
                except Exception as e:
                    print(f"Error in forward pass: {e}")
                    return

            target = pred.clone()
            Q_new = reward[0][0]
            if not done:
                try:
                    with torch.no_grad():  # Using no_grad in inference to save memory
                        Q_new = reward[0][0] + self.gamma * torch.max(self.model(next_state))
                except Exception as e:
                    print(f"Error in calculating Q_new: {e}")
                    return

            target[0][torch.argmax(action).item()] = Q_new

            # Q_new = r + y * max(next_predicted Q value)
            self.optimizer.zero_grad()
            try:
                loss = self.criterion(target, pred)
                print(f"Loss value: {loss.item()}")  # Track the loss value
                # Check for NaN or Inf in loss
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    print("Detected NaN or Inf in loss.")
                    return
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Clip gradients
                self.optimizer.step()

                # Free memory after the step
                del state, next_state, action, reward, pred, target, loss
                torch.cuda.empty_cache()  # Free up GPU memory

            except Exception as e:
                print(f"Error in backpropagation: {e}")
                return

        except Exception as e:
            print(f"Error in train_step: {e}")
