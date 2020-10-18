"""
This file starts polling the server every minute and trains the model if it finds the next model
"""

import time
from dotenv import load_dotenv

from client import client
from azure_blob import AzureBlob

load_dotenv()

epoch = 0
EPOCH_COUNT = 10

container_name = 'loans-c'
cloud_helper = AzureBlob()
client = client(cloud_helper, container_name)

while epoch < EPOCH_COUNT:
    epoch = client.poll_server(epoch)
    print('ab to sleep' + str(epoch))
    time.sleep(5)


