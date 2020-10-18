import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from dotenv import load_dotenv

from azure.azure_blob import AzureBlob
from azure.utils import blob_pre_model_name, server_local_pre_model_name, blob_post_model_name, server_local_post_model_name

class NN(nn.Module):
    def __init__(self, input_features=11, layer1=20, layer2=20, out_features=2):
        """Initialize the model for loan prediction"""
        super().__init__()
        self.fc1 = nn.Linear(input_features, layer1)
        self.fc2 = nn.Linear(layer1, layer2)
        self.out = nn.Linear(layer2, out_features)
        
    def forward(self, x):
        """Forward pass with 11 input features"""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

load_dotenv()

epoch = 0
EPOCH_COUNT = 10

federated_list = [
    {
        "company_a": {
        "cloud": AzureBlob,
        "container": "loans-a",
        "env_key": "something"
    }}
]

def publish_state_dict(model_state_dict, epoch):
    """ Saves model locally then pushes to all databases """
    torch.save(model_state_dict, server_local_pre_model_name(epoch))

    for entity in federated_list: 
        cloud_helper = list(entity.values())[0]['cloud']()
        container = list(entity.values())[0]['container']
        cloud_helper.upload_to_blob_storage(server_local_pre_model_name(epoch), container, blob_pre_model_name(epoch))

def poll_clients(epoch, remaining):
    """ Looking for post_models from clients """
    for entity in remaining:
        cloud_helper = list(entity.values())[0]['cloud']()
        container = list(entity.values())[0]['container']
        if (cloud_helper.check_for_file(container, blob_post_model_name(epoch))):
            cloud_helper.download_from_blob_storage(server_local_post_model_name(epoch, container), container, blob_post_model_name(epoch))
            remaining.remove(entity)
    return remaining

def federated_averaging(epoch):
    result = None
    for entity in federated_list:
        container = list(entity.values())[0]['container']

        if not result: # first entity
            result = torch.load(server_local_post_model_name(epoch, container))
            continue
        else: # other entities
            state_dict = torch.load(server_local_post_model_name(epoch, container))

        for param in result:
            result[param] = result[param] + state_dict[param]
    
    for param in result:
        result[param] = result[param] / len(federated_list)
    
    return result


remaining = federated_list.copy()

torch.manual_seed(0)
model = NN()
publish_state_dict(model.state_dict(), epoch)

while epoch < EPOCH_COUNT:
    remaining = poll_clients(epoch, remaining)

    if not remaining:
        updated_model = federated_averaging(epoch)
        epoch += 1
        publish_state_dict(updated_model, epoch)
        remaining = federated_list.copy()
    else:
        print('ab to sleep')
        time.sleep(5)


