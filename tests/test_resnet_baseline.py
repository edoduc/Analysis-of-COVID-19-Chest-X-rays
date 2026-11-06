from src.constants import CLASS_NAMES
from src.train_resnet_baseline import build_model


def test_baseline_freezes_backbone_and_trains_classification_head():
    model = build_model(pretrained=False)

    assert model.fc.out_features == len(CLASS_NAMES)
    assert all(not parameter.requires_grad for parameter in model.layer1.parameters())
    assert all(parameter.requires_grad for parameter in model.fc.parameters())