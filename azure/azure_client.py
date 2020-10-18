import torch
import torch.nn as nn
import torch.nn.functional as F

# this will be loaded in somehow later, 
# maybe pickle?
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

container_name = 'loans'
# TODO: standardize filenames

class azure_client():
    """ Generalizable client for Azure-based models""" 

    def check_for_new_model(ab, epoch):
        """ Checks Azure Blob Storage whether next model is uploaded and downloads if it is """ 
        new_model_filepath = 'server_model_' + str(epoch) + '.pth'
        local_filepath = './models/server_model_' + str(epoch) + '.pth'

        if ab.check_for_file(container_name, new_model_filepath):
            ab.download_from_blob_storage(local_filepath, container_name, new_model_filepath)
            return True

    def upload_trained_model(ab, epoch, model):
        """ This will upload the trained model to Azure blob"""

    def poll_server(ab, epoch):
        """ 
        - Periodically polled to download new models 
        - Then train the model
        - Then upload the trained model
        """ 
        # This automatically downloads from server if new model exists
        if(azure_client.check_for_new_model(ab, epoch)):
            print('nice now to train some stuff')
            
            # This loads it and trains
            local_filepath = './models/server_model_' + str(epoch) + '.pth'
            model = NN()
            model.load_state_dict(local_filepath)

            model.train()
            
            # Then upload 
            azure_client.

            # increment epoch
            epoch += 1
        