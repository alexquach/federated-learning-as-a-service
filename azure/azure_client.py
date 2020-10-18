import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

from utils import blob_pre_model_name, local_pre_model_name, blob_post_model_name, local_post_model_name

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

def preprocess_df(dataframe, isTest=False):
    """Preprocess a dataframe, unique to the loan_prediction dataset"""
    #perform deep copy, fixes self assignment bug:
    #https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    df = dataframe.copy(deep=True)
    
    # null_df = np.sum(df.isnull())
    # print(null_df) 
    # print(f"\nTotal null values: {np.sum(null_df)}") #get total number of null values
    ### remove all rows with null values
    df = df.dropna(how='any',axis=0) 
    del df['Loan_ID'] #remove Loan_ID (irrelevant)

    # convert to binary variables

    ##----------------------------------------------------------------------------
    #### ----------------------------------Table----------------------------------
    ##----------------------------------------------------------------------------

    #> ----Gender---
    ## - Male: 0
    ## - Female: 1
    df.loc[(df.Gender == 'Male'),'Gender']=0
    df.loc[(df.Gender == 'Female'),'Gender']=1

    #> ----Married---
    ## - No: 0
    ## - Yes: 1
    df.loc[(df.Married == 'Yes'),'Married']=0
    df.loc[(df.Married == 'No'),'Married']=1

    #> ----Education---
    ## - Not Graduate: 0
    ## - Graduate: 1
    df.loc[(df.Education == 'Not Graduate'),'Education']=0
    df.loc[(df.Education == 'Graduate'),'Education']=1

    #> ----Self_Employed---
    ## - No: 0
    ## - Yes: 1
    df.loc[(df.Self_Employed == 'No'),'Self_Employed']=0
    df.loc[(df.Self_Employed == 'Yes'),'Self_Employed']=1


    #> ----Property_area---
    ## - Rural: 0
    ## - Urban: 1
    ## - Semiurban: 2
    df.loc[(df.Property_Area == 'Rural'),'Property_Area']=0
    df.loc[(df.Property_Area == 'Urban'),'Property_Area']=1
    df.loc[(df.Property_Area == 'Semiurban'),'Property_Area']=2
    
    
    #> ----Loan_Status--- (ONLY for Training set)
    ## - No: 0
    ## - Yes: 1
    if(not isTest):
        df.loc[(df.Loan_Status == 'N'),'Loan_Status']=0
        df.loc[(df.Loan_Status == 'Y'),'Loan_Status']=1

    #> -----Dependents-----
    #set max as 
    df.loc[(df.Dependents == '3+'), 'Dependents'] = 3
    ##----------------------------------------------------------------------------
    #### ----------------------------------Table----------------------------------
    ##----------------------------------------------------------------------------

    #!!! Typecase to float (for tensors below)
    df = df.astype(float)
    
    return df

#TODO: @Alex: update azure_client to generalize to GCP Client

class azure_client():
    """ Generalizable client for Azure-based models""" 
    def __init__(self, ab, container_name): 
        self.ab = ab #need to change this such that it can also implement GCPBlob.
        self.container_name = container_name

    def _check_for_new_model(self, epoch):
        """ Checks Azure Blob Storage whether next model is uploaded and downloads if it is """ 
        if self.ab.check_for_file(self.container_name, blob_pre_model_name(epoch)):
            self.ab.download_from_blob_storage(local_pre_model_name(epoch), self.container_name, blob_pre_model_name(epoch))
            return True

    def _load_data(self):
        """ Reading data locally """
        train_df = pd.read_csv("./data/archive/train.csv")
        test_df = pd.read_csv("./data/archive/test.csv")

        train_df = preprocess_df(train_df)
        test_df = preprocess_df(test_df, isTest=True)

        X_train = train_df.drop('Loan_Status',axis=1).values
        y_train = train_df['Loan_Status'].values
        X_test = test_df.values

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #Enable cuda if available

        X_train = torch.FloatTensor(X_train).to(device)
        X_test = torch.FloatTensor(X_test).to(device)
        y_train = torch.LongTensor(y_train).to(device)

        return X_train, y_train, device

    def _train(self, model, X_train, y_train, train_epochs=int(1e2), print_every=100, epsilon=0.5):
        """
        Train the model.
         @Param:
        1. epochs - number of training iterations.
        2. print_every - for visual purposes (set to None to ignore), outputs loss
        3. epsilon - threshold to break training.
        """
        start_time = time.time() #set start time
        losses = [] #plot

        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005) #adam works well for this.
        
        for i in range(1, train_epochs+1):
            y_pred = model(X_train)
            loss = loss_function(y_pred, y_train)
            losses.append(loss)
            
            if(loss.item() <= epsilon):
                print(f"\nCONVERGED at epoch {i} - loss : {loss.item()}")
                break #converged
            
            if(print_every is not None and i%print_every == 1):
                print(f"Epoch {i} - loss : {loss.item()}")
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print("total training time (sec):", time.time()-start_time)
        return model, losses

    def _upload_trained_model(self, epoch, model):
        """ This will upload the trained model to Azure blob"""
        torch.save(model.state_dict(), local_post_model_name(epoch))
        self.ab.upload_to_blob_storage(local_post_model_name(epoch), self.container_name, blob_post_model_name(epoch))

    def poll_server(self, epoch):
        """ 
        - Periodically polled to download new models 
        - Then train the model
        - Then upload the trained model
        """ 
        # This automatically downloads from server if new model exists
        if(self._check_for_new_model(epoch)):
            
            # Get Train data
            X_train, y_train, device = self._load_data()

            # Load downloaded model and train it further
            torch.manual_seed(0)
            model = NN()
            model = model.to(device)

            state_dict = torch.load(local_pre_model_name(epoch))
            model.load_state_dict(state_dict) 

            # Train model further
            updated_model, loss = self._train(model, X_train, y_train)
            
            # Then upload 
            self._upload_trained_model(epoch, updated_model)

            # increment epoch
            epoch += 1
        return epoch
        