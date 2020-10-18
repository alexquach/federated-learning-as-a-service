# for testing purposes only

from gcp_blob import GCPBlob
import sys
import time
if __name__ == "__main__":
    start_time = time.time()
    gcp = GCPBlob()
    gcp.upload_to_blob_storage(sys.argv[1], sys.argv[2], sys.argv[3])

    end_time = time.time()
    print("Time taken for operation (sec):", end_time - start_time)