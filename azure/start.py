"""
This file starts polling the server every minute and trains the model if it finds the next model
"""

import time
from dotenv import load_dotenv

from azure_client import azure_client
from azure_blob import AzureBlob

load_dotenv()

epoch = 0
EPOCH_COUNT = 10

container_name = 'loans-a'
ab = AzureBlob()
azure_client = azure_client(ab, container_name)

while epoch < EPOCH_COUNT:
    epoch = azure_client.poll_server(epoch)
    print('ab to sleep')
    time.sleep(5)


