import torch
import pytest
from src.models.model import MyAwesomeConvNext


class TestClass:
    model = MyAwesomeConvNext("convnext_atto", False, 3, 10)

    def test_model(self):
        assert self.model(torch.rand(1, 3, 224, 224)).shape == (1, 10,),\
            "Model gives wrong dimensions"

    def test_error_on_4D_shape(self):
        with pytest.raises(ValueError, match="Expected input to a 4D tensor"):
            self.model(torch.rand(1, 224, 224)).shape == (1, 10)

    def test_error_on_shape_1(self):
        with pytest.raises(
                ValueError, match="Expected dim 1 of sample to have shape {3}"):
            self.model(torch.rand(1, 6, 224, 224)).shape == (1, 10)

    def test_error_on_shape_2(self):
        with pytest.raises(
                ValueError, match="Expected dim 2 of sample to have shape {224}"):
            self.model(torch.rand(1, 3, 6, 224)).shape == (1, 10)

    def test_error_on_shape_3(self):
        with pytest.raises(
                ValueError, match="Expected dim 3 of sample to have shape {224}"):
            self.model(torch.rand(1, 3, 224, 6)).shape == (1, 10)
