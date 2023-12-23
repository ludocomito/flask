from flask import Flask, request, jsonify
import torch
import numpy as np
import torch.nn as nn


# Model definition
class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        
        # policy_net (mlp_extractor)
        self.policy_net_0 = nn.Linear(48, 1024)
        self.policy_net_2 = nn.Linear(1024, 512)
        self.policy_net_4 = nn.Linear(512, 256)
        self.action_net = nn.Linear(256, 5)
        
        # value_net (mlp_extractor)
        self.value_net_0 = nn.Linear(48, 2048)
        self.value_net_2 = nn.Linear(2048, 1024)
        self.value_net_4 = nn.Linear(1024, 512)
        self.value_net = nn.Linear(512, 1)

        # Relu activation function
        self.relu = nn.ReLU()
        
    
    def forward(self, x):
        # policy_net (mlp_extractor)
        x_ = x
        x = self.relu(self.policy_net_0(x))        
        x = self.relu(self.policy_net_2(x))        
        x = self.relu(self.policy_net_4(x))        
        action = self.action_net(x)

        # value_net (mlp_extractor)
        x = self.relu(self.value_net_0(x_))
        x = self.relu(self.value_net_2(x))
        x = self.relu(self.value_net_4(x))
        value = self.value_net(x)
       
        return action, value

# Instiate the model
load_path = 'policy.pth'
model = Policy()

state_dict = torch.load(load_path, map_location=torch.device('cpu'))

model = Policy()
model.eval()

# Assign the weights using the state_dict
model.policy_net_0.weight.data = state_dict['mlp_extractor.policy_net.0.weight']
model.policy_net_0.bias.data = state_dict['mlp_extractor.policy_net.0.bias']
model.policy_net_2.weight.data = state_dict['mlp_extractor.policy_net.2.weight']
model.policy_net_2.bias.data = state_dict['mlp_extractor.policy_net.2.bias']
model.policy_net_4.weight.data = state_dict['mlp_extractor.policy_net.4.weight']
model.policy_net_4.bias.data = state_dict['mlp_extractor.policy_net.4.bias']
model.action_net.weight.data = state_dict['action_net.weight']
model.action_net.bias.data = state_dict['action_net.bias']

model.value_net_0.weight.data = state_dict['mlp_extractor.value_net.0.weight']
model.value_net_0.bias.data = state_dict['mlp_extractor.value_net.0.bias']
model.value_net_2.weight.data = state_dict['mlp_extractor.value_net.2.weight']
model.value_net_2.bias.data = state_dict['mlp_extractor.value_net.2.bias']
model.value_net_4.weight.data = state_dict['mlp_extractor.value_net.4.weight']
model.value_net_4.bias.data = state_dict['mlp_extractor.value_net.4.bias']
model.value_net.weight.data = state_dict['value_net.weight']
model.value_net.bias.data = state_dict['value_net.bias']

print("All the layers imported successfully")

# Flask server
app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Parse the JSON to get the tensor list
        data = request.get_json()
        tensor_list = data['tensor']

        # Convert the list back to a tensor
        tensor = torch.tensor(tensor_list)

        # Your model inference here
        with torch.no_grad():
            action, value = model(tensor)

        # Send back the result
        return jsonify({'action': action.numpy().tolist(), 'value': value.numpy().tolist()})

if __name__ == '__main__':
    app.run()
