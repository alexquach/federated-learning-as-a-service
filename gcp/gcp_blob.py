import sys
import time

from google.cloud import storage

# Functionality:
#> 1. Update & View metadata to determine (3) and (4).
#> 2. Train model based on model.pth file (3).
#> 3. Download a file.
#> 4. Upload a file, enforcing conditions from (1).

class GCPBlob():
    def __init__(self):
        """ Create a GCP client stoage object that can perform CRUD operations. """
        self.client = storage.Client()
    
    def get_container_client(self, bucket_Name):
        """ Returns a bucket (container in Azure) from the specified service account. """
        return self.client.bucket(bucket_Name)
    
    def get_blob_client(self, bucket_Name, blob_name):
        """ Get a specified blob from a bucket. Throws exception if not found. """
        obj = None
        try:
            bucket = self.get_container_client(bucket_Name)
            obj = bucket.get_blob(blob_name)
        except Exception as e:
            print(e)
        
        return obj
    
    def upload_to_blob_storage(self, local_filepath, bucket_Name, blob_name):
        """ Uploads file at `local_filepath` to the blob at `blob_name` """
        #Example: upload_to_blob_storage('data/models/policy.pth', 'loan_company_a', 'policy.pth')
        try: 
            bucket = self.get_container_client(bucket_Name)
            if(bucket is None):
                raise ValueError("Bucket object can't be null")

            blob = bucket.blob(blob_name)
            blob.upload_from_filename(local_filepath)

        except Exception as e:
            print(e)

        return
    
    def download_from_blob_storage(self, local_filepath, bucket_Name, blob_name):
        """ Downloads file at `blob_name` to the local path at `local_filepath` """
        #Example: download_from_blob_storage('data/models/policy.pth', 'loan_company_a', 'policy.pth')
        try:
            blob = self.get_blob_client(bucket_Name, blob_name)
            if(blob is None):
                raise ValueError("Blob object can't be null")

            blob.download_to_filename(local_filepath)

        except Exception as e:
            print(e)

        return
    
    def check_for_file(self, bucket_name, blob_name):
        """ Checks whether a Bucket `bucket_name` has the Blob with filename `blob_name` """
        #Example: check_for_file('loan_company_a', 'policy.pth')
        found = False
        try:
            found = blob_name in [blob.name for blob in self.get_container_client(bucket_name).list_blobs()]
        except Exception as e:
            print(e)

        return found