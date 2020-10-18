import os
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, __version__

class AzureBlob():
    def __init__(self):
        """ Store BlobServiceClient """
        connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
        self.client = BlobServiceClient.from_connection_string(connect_str)


    def get_container_client(self, container_name):
        """ Container Client """ 
        return self.client.get_container_client(container_name)


    def get_blob_client(self, container_name, blob_name):
        """ Blob Client """
        return self.client.get_blob_client(container=container_name, blob=blob_name)


    def upload_to_blob_storage(self, local_filepath, container_name, blob_name):
        """ Uploads file at `local_filepath` to the blob at `blob_name` """
        with open(local_filepath, "rb") as data:
            try: 
                self.get_blob_client(container_name, blob_name).upload_blob(data)
            except:
                pass
        return


    def download_from_blob_storage(self, local_filepath, container_name, blob_name):
        """ Downloads file at `blob_name` to the local path at `local_filepath` """
        with open(local_filepath, "wb") as download_file:
            try:
                download_file.write(self.get_blob_client(container_name, blob_name).download_blob().readall())
            except:
                pass
        return
        
    
    def check_for_file(self, container_name, blob_name):
        """ Checks whether a Container `container_name` has the Blob with filename `blob_name` """
        for blob in self.get_container_client(container_name).list_blobs():
            if blob_name == blob.name:
                return True
        return False
