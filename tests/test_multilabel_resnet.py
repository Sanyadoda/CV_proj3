from vision.multilabel_resnet import MultilabelResNet18
from tests.model_test_utils import extract_model_layers


def test_multilabel_resnet():
    """
    Tests the transforms using output from disk
    """
    this_res_net = MultilabelResNet18()

    (
        _,
        output_dim,
        _,
        num_params_grad,
        num_params_nograd,
    ) = extract_model_layers(this_res_net)
   
    assert output_dim == 7
    assert num_params_grad < 10000
    assert num_params_nograd > 1e7
