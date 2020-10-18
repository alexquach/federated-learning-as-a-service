import os
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

class client():
    """ Generalizable client for Azure-based models""" 
    def __init__(self, cloud_helper, container_name): 
        self.cloud_helper = cloud_helper
        self.container_name = container_name

    def _check_for_new_pre_model(self, epoch):
        """ Checks Azure Blob Storage whether next model is uploaded and downloads if it is """ 
        # check if pre_model_0 is in container `loans-a`
        if self.cloud_helper.check_for_file(self.container_name, blob_pre_model_name(epoch)):
            # downloads locally 
            self.cloud_helper.download_from_blob_storage(local_pre_model_name(epoch), self.container_name, blob_pre_model_name(epoch), delete_local_name=local_pre_model_name(epoch-1))
            return True

    def _load_data(self):
        """ Reading data locally """
        train_df = pd.read_csv("./data/{0}/train.csv".format(self.container_name))
        test_df = pd.read_csv("./data/{0}/test.csv".format(self.container_name))

        train_df = preprocess_df(train_df)
        test_df = preprocess_df(test_df)

        X_train = train_df.drop('Loan_Status',axis=1).values
        y_train = train_df['Loan_Status'].values
        X_test = test_df.drop('Loan_Status',axis=1).values
        y_test = test_df['Loan_Status'].values

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #Enable cuda if available

        X_train = torch.FloatTensor(X_train).to(device)
        X_test = torch.FloatTensor(X_test).to(device)
        y_train = torch.LongTensor(y_train).to(device)
        y_test = torch.LongTensor(y_test).to(device)

        return X_train, y_train, X_test, y_test, device

    def _train(self, model, X_train, y_train, X_test, y_test, train_epochs=int(1e3), print_every=100, epsilon=0.5):
        """
        Train the model.
         @Param:
        1. epochs - number of training iterations.
        2. print_every - for visual purposes (set to None to ignore), outputs loss
        3. epsilon - threshold to break training.
        """
        start_time = time.time() #set start time
        losses = [] #plot
        accuracies = []

        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005) #adam works well for this.
        
        for i in range(1, train_epochs+1):
            y_pred = model(X_train)
            loss = loss_function(y_pred, y_train)
            losses.append(loss)
            
            y_pred = model(X_test)
            y_pred = torch.Tensor([0 if x[0] > x[1] else 1 for x in y_pred])
            accuracy = (y_pred == y_test).float().sum() / len(y_pred)
            accuracies.append(accuracy)
            
            if(loss.item() <= epsilon):
                print(f"\nCONVERGED at epoch {i} - loss : {loss.item()}")
                break #converged
            
            if(print_every is not None and i%print_every == 1):
                print(f"Epoch {i} - loss : {loss.item()} - accuracy : {accuracy.item()}")
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        
        print("total training time (sec):", time.time()-start_time)
        return model, losses

    def _upload_trained_model(self, epoch, model):
        """ This will upload the trained model to Azure blob"""
        # saves locally
        torch.save(model.state_dict(), local_post_model_name(epoch, self.container_name))

        try:
            os.remove(local_post_model_name(epoch-1, self.container_name))
        except:
            pass

        # pushes to container
        self.cloud_helper.upload_to_blob_storage(local_post_model_name(epoch, self.container_name), self.container_name, blob_post_model_name(epoch), delete_blob_name=blob_post_model_name(epoch-1))

    def poll_server(self, epoch):
        """ 
        - Periodically polled to download new models 
        - Then train the model
        - Then upload the trained model
        """ 
        # This automatically downloads from server if new model exists
        if(self._check_for_new_pre_model(epoch)):
            
            # Get Train data
            X_train, y_train, X_test, y_test, device = self._load_data()

            # Load downloaded model and train it further
            torch.manual_seed(0)
            model = NN()
            model = model.to(device)

            print(local_pre_model_name(epoch))
            state_dict = torch.load(local_pre_model_name(epoch))
            model.load_state_dict(state_dict) 

            # Train model further
            updated_model, loss = self._train(model, X_train, y_train, X_test, y_test)
            
            # Then upload post_model_0
            self._upload_trained_model(epoch, updated_model)

            # increment epoch
            epoch += 1
        return epoch
        
    def solo_train(self):
        X_train, y_train, X_test, y_test, device = self._load_data()

        model = NN()
        model = model.to(device)

        updated_model, loss = self._train(model, X_train, y_train, X_test, y_test)