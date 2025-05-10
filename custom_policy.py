import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.dqn.policies import QNetwork, DQNPolicy

class CustomFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for DQN with attention mechanism
    """
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        n_input_features = observation_space.shape[0]
        
        # Deeper network with residual connections
        self.feature_net = nn.Sequential(
            nn.Linear(n_input_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),
            
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),
            
            nn.Linear(256, features_dim),
            nn.ReLU()
        )
        
        # Self-attention layer
        self.attention = nn.Sequential(
            nn.Linear(features_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, observations):
        # # Ensure consistent dtype
        # observations = observations.to(th.float32)
        
        # Convert to float32 explicitly
        if not isinstance(observations, th.Tensor):
            observations = th.tensor(observations, dtype=th.float32)
        else:
            observations = observations.to(th.float32)
        
        # Handle both single observations and batches
        single_input = observations.dim() == 1
        if single_input:
            observations = observations.unsqueeze(0)  # Add batch dimension
            
        features = self.feature_net(observations)
        
        # Apply attention (reshape for attention calculation)
        attention_weights = self.attention(features.unsqueeze(1))
        attended_features = attention_weights * features.unsqueeze(1)
        
        result = attended_features.squeeze(1)
        
        # Remove batch dimension if input was a single observation
        if single_input:
            result = result.squeeze(0)
            
        return result

class CustomQNetwork(QNetwork):
    """
    Custom Q-Network with dueling architecture
    """
    def __init__(self, observation_space, action_space, features_extractor=None, 
                 features_dim=128, dueling=True, **kwargs):
        super().__init__(observation_space, action_space, features_extractor, features_dim)
        
        self.dueling = dueling
        n_actions = action_space.n
        
        # Get the actual input dimension
        if features_extractor is None:
            input_dim = observation_space.shape[0]  # Use observation dimension directly
        else:
            input_dim = features_dim  # Use the output dimension of the feature extractor
        
        # Advantage stream
        self.advantage_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )
        
        # Value stream (only used if dueling)
        if self.dueling:
            self.value_net = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )
    
    def forward(self, obs):
        # Convert to float32 explicitly
        if not isinstance(obs, th.Tensor):
            obs = th.tensor(obs, dtype=th.float32)
        else:
            obs = obs.to(th.float32)
        
        # Handle both single observations and batches
        single_input = obs.dim() == 1
        if single_input:
            obs = obs.unsqueeze(0)  # Add batch dimension
            
        # Extract features
        if self.features_extractor is None:
            features = obs
        else:
            features = self.features_extractor(obs)
        
        # Print shape for debugging
        # print(f"Features shape: {features.shape}")
        
        # Ensure features are float32
        features = features.to(th.float32)
        
        if self.dueling:
            advantage = self.advantage_net(features)
            value = self.value_net(features)
            # Combine value and advantage using dueling formula
            q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        else:
            q_values = self.advantage_net(features)
        
        # Remove batch dimension if input was a single observation
        if single_input:
            q_values = q_values.squeeze(0)
            
        return q_values

class CustomDQNPolicy(DQNPolicy):
    """
    Custom DQN policy using our enhanced network architecture
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, 
                         features_extractor_class=CustomFeatureExtractor,
                         features_extractor_kwargs=dict(features_dim=128),
                         **kwargs)
        
    def make_q_net(self):
        return CustomQNetwork(
            self.observation_space,
            self.action_space,
            self.features_extractor,
            dueling=True
        )
        
    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        """
        Override predict method to ensure proper dtype handling
        """
        if not isinstance(observation, th.Tensor):
            observation = th.tensor(observation, dtype=th.float32)
        else:
            observation = observation.to(th.float32)
            
        return super().predict(observation, state, episode_start, deterministic)
