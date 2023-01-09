# import torch
# from src.models.model import MyAwesomeModel
# import pytest
#
#
# class TestClass:
#     model = MyAwesomeModel()
#
#     def test_model(self):
#         assert self.model(torch.rand(1, 1, 28, 28)).shape == (1, 10)
#
#     def test_error_on_wrong_shape1(self):
#         with pytest.raises(ValueError, match="Expected input to a 4D tensor"):
#             self.model(torch.randn(1, 2, 3))
#
#     def test_error_on_wrong_shape2(self):
#         with pytest.raises(
#             ValueError, match="Expected each sample to have shape 1, 28, 28"
#         ):
#             self.model(torch.randn(1, 2, 28, 28))
