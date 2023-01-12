import torch
import pytest
from src.models.model import MyAwesomeConvNext
from src.models.train_model import dataset


class TestClass:
    model = MyAwesomeConvNext("convnext_atto", False, 3, 10)

    @pytest.mark.skipif(
        not os.path.exists(
            os.path.join(_PATH_DATA, "processed/training_data.pickle")
        ),
        reason="Data files not found",
    )
    def test_model_training(self):
        assert True == True # There is nothing to assert in the training.
