from aide.core.trainable import TrainableModel


def test_trainable_model_abstract_methods():

    try:

        class DummyTrainableModel(TrainableModel):  # type: ignore
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def forward(self, *args, **kwargs):
                return super().forward(*args, **kwargs)  # type: ignore

            def training_step(self, *args, **kwargs):
                return super().training_step(*args, **kwargs)  # type: ignore

            def configure_optimizers(self, *args, **kwargs):
                return super().configure_optimizers(*args, **kwargs)  # type: ignore
    except TypeError as e:
        assert "Can't instantiate abstract class" in str(e)
