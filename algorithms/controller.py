__all__ = ['Controller']


class Controller(object):

    def __init__(self):
        raise NotImplementedError

    @staticmethod
    def get_modules(*args, **kwargs):
        raise NotImplementedError

    def state_dict(self):
        raise NotImplementedError

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def step_pre_training(self, *args, **kwargs):
        """Update the quantization hyper-parameters before the training step of the current epoch."""
        raise NotImplementedError

    def step_pre_validation(self, *args, **kwargs):
        """Update the quantization hyper-parameters before the validation step of the current epoch."""
        raise NotImplementedError
