
import os
import torch
import torch.utils.data
import numpy as np

from six import BytesIO
from lstm_model import LSTMPredictor


NP_CONTENT_TYPE = 'application/x-npy'


def model_fn(model_dir):
    # First, load the parameters used to create the model.
    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print('model_info: {}'.format(model_info))

    # Determine the device and construct the model.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = LSTMPredictor(
        model_info['input_dim'],
        model_info['hidden_dim'],
        model_info['output_dim']
    )

    # Load the store model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    model.to(device).eval()

    print('Done loading model.')
    return model


def input_fn(serialized_input_data, content_type):
    print('Deserializing the input data.')
    
    if content_type == NP_CONTENT_TYPE:
        stream = BytesIO(serialized_input_data)
        return np.load(stream)
    raise Exception(
        'Requested unsupported ContentType in content_type: ' + content_type
    )


def output_fn(prediction_output, accept):
    print('Serializing the generated output.')
    
    if accept == NP_CONTENT_TYPE:
        stream = BytesIO()
        np.save(stream, prediction_output)
        return stream.getvalue(), accept
    
    raise Exception('Requested unsupported ContentType in Accept: ' + accept)


def predict_fn(input_data, model):
    print('Inferring reveunue for input data.')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('input_data::', input_data)
    
    data = torch.from_numpy(input_data.astype('float32'))
    data = data.to(device)

    # Make sure to put the model into evaluation mode
    model.eval()

    with torch.no_grad():
        output = model(data)

    result = torch.round(output).cpu().detach().numpy()

    return result
