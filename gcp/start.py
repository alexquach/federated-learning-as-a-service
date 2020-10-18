"""
This file starts polling the server every minute and trains the model if it finds the next model
"""

import time
from dotenv import load_dotenv
import sys
sys.path.append("../azure")
from gcp_blob import GCPBlob
from azure_client import azure_client
from azure_blob import AzureBlob

load_dotenv()

epoch = 0
EPOCH_COUNT = 10

container_name = 'company_a_loan'
ab = GCPBlob()
azure_client = azure_client(ab, container_name)

while epoch < EPOCH_COUNT:
    epoch = azure_client.poll_server(epoch)
    print(f'Epoch: {epoch} ab to sleep')
    time.sleep(5)


