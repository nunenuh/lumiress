import copy
from unittest.mock import patch

import torch

from lumiress.models.mirnetv2_arch import MIRNetV2
from lumiress.models.mirnetv2_builder import (_base_parameters, build_model,
                                              get_param, get_weight)


def test_get_param():
    # Testing when name is in _task_name and name is "super_resolution"
    name = "super_resolution"
    expected_output = copy.deepcopy(_base_parameters)
    expected_output["scale"] = 4
    assert get_param(name) == expected_output

    # Testing when name is in _task_name but name is not "super_resolution"
    name = "other_task"
    expected_output = copy.deepcopy(_base_parameters)
    assert get_param(name) == expected_output

    # Testing when name is not in _task_name
    name = "unknown_task"
    expected_output = copy.deepcopy(_base_parameters)
    assert get_param(name) == expected_output


from unittest.mock import patch


@patch("torch.load")
@patch("lumiress.io.downloader._download_weight")
@patch("pathlib.Path.exists")
@patch("pathlib.Path.is_file")
def test_get_weight(mock_is_file, mock_exists, mock_download_weight, mock_torch_load):
    # Mock the Path methods
    mock_exists.return_value = True
    mock_is_file.return_value = True

    # Mock the torch.load function
    weight = "some_weight"
    mock_torch_load.return_value = weight

    # Test when the file exists
    assert get_weight("real_denoising") == weight
    mock_torch_load.assert_called_once()

    # # Now test when the file does not exist
    # mock_exists.return_value = False
    # mock_is_file.return_value = False
    # mock_torch_load.reset_mock()  # reset call count

    # assert get_weight('real_denoising') == weight
    # mock_download_weight.assert_called_once_with('real_denoising')
    # mock_torch_load.assert_called_once()


def test_model_creation():
    # Test if the function creates a model object
    model = build_model("real_denoising")
    assert isinstance(model, MIRNetV2)


def test_model_state_dict():
    # Test if the function loads the correct state_dict for the model
    model = build_model("real_denoising")
    checkpoint = get_weight("real_denoising")
    state_dict = checkpoint["params"]
    model_dict = model.state_dict()
    # Iterate over all keys in the state dictionary
    for key in state_dict.keys():
        # Check if the tensors stored under each key are close
        assert torch.allclose(model_dict[key], state_dict[key])


def test_model_eval_mode():
    # Test if the function sets the model to evaluation mode
    model = build_model("lowlight_enhancement")
    assert not model.training
