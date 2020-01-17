
import os
import torch
import torch.utils.data

from model import LSTMPredictor


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
        model_info['embedding_dim'],
        model_info['hidden_dim']
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
    if content_type == 'text/plain':
        data = serialized_input_data.decode('utf-8')
        return data
    raise Exception(
        'Requested unsupported ContentType in content_type: ' + content_type
    )


def output_fn(prediction_output, accept):
    print('Serializing the generated output.')
    return str(prediction_output)


def predict_fn(input_data, model):
    print('Inferring reveunue for input data.')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = input_data
    data = data.to(device)

    # Make sure to put the model into evaluation mode
    model.eval()

    # TODO: Compute the result of applying the model to the input data.
    # The variable `result` should be a numpy array which contains a single
    # integer which is either 1 or 0
    with torch.no_grad():
        output = model(data)

    result = torch.round(output).cpu().detach().numpy()

    return result