"""
This file starts polling the server every minute and trains the model if it finds the next model
"""

import time
import sys
from dotenv import load_dotenv

from client import client
from clients.azure_blob import AzureBlob
from clients.gcp_blob import GCPBlob

load_dotenv()

if __name__ == "__main__":
    epoch = 0
    EPOCH_COUNT = 10

    if (sys.argv[1] == 'azure'):
        cloud_helper = AzureBlob()
    if (sys.argv[1] == 'gcp'):
        cloud_helper = GCPBlob()

    if (sys.argv[2]):
        container_name = sys.argv[2]
    else:
        container_name = 'loans-a'

    if (sys.argv[3]):
        epoch = int(sys.argv[3])

    client = client(cloud_helper, container_name)

    if (sys.argv[4]):
        client.solo_train()
    else:

        while epoch < EPOCH_COUNT:
            epoch = client.poll_server(epoch)
            print('ab to sleep' + str(epoch))
            time.sleep(5)

