def blob_pre_model_name(epoch):
    return 'pre_model_' + str(epoch) + '.pth'
def local_pre_model_name(epoch):
    return './models/pre_model_' + str(epoch) + '.pth'
def blob_post_model_name(epoch):
    return 'post_model_' + str(epoch) + '.pth'
def local_post_model_name(epoch):
    return './models/post_model_' + str(epoch) + '.pth'
def server_local_pre_model_name(epoch):
    return './server_models/pre_model_' + str(epoch) + '.pth'
def server_local_post_model_name(epoch, container):
    return './server_models/' + container + '/post_model_' + str(epoch) + '.pth'