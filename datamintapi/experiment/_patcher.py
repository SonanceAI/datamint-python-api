from unittest.mock import patch
import importlib
from typing import Sequence
import logging

_LOGGER = logging.getLogger(__name__)


def _is_iterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False


class Wrapper:
    def __init__(self,
                 target: str,
                 experiment,
                 cb_before: Sequence[callable] | callable = None,
                 cb_after: Sequence[callable] | callable = None) -> None:
        self.experiment = experiment
        self.cb_before = cb_before if cb_before is not None else []
        if not _is_iterable(self.cb_before):
            self.cb_before = [self.cb_before]
        self.cb_after = cb_after if cb_after is not None else []
        if not _is_iterable(self.cb_after):
            self.cb_after = [self.cb_after]
        self.target = target
        self._patch()

    def _patch(self):
        def _callback(*args, **kwargs):
            for cb in self.cb_before:
                cb(self.experiment, original, args, kwargs)

            try:
                return_value = original(*args, **kwargs)
            except Exception as exception:
                # We are assuming the patched function does not return an exception.
                return_value = exception

            for cb in self.cb_after:
                cb(self.experiment, original, args, kwargs, return_value)

            if isinstance(return_value, Exception):
                raise return_value

            return return_value

        original = get_function_from_string(self.target)
        # Patch the original function with the callback
        self.patcher = patch(self.target, new=_callback)

    def start(self):
        self.patcher.start()

    def stop(self):
        self.patcher.stop()


def _backward_callback(experiment, original_obj, func_args, func_kwargs):
    """
    This function is a wrapper for the backward method of the Pytorch Tensor class.
    """
    loss = func_args[0]
    print(f"Logging the backward method of {original_obj.__name__} with loss={loss}")


def get_function_from_string(target: str):
    target_spl = target.split('.')
    for i in range(len(target_spl)):
        module_name = '.'.join(target_spl[:-i-1])
        function_name = '.'.join(target_spl[-i-1:])
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        break
    else:
        raise ModuleNotFoundError(f"Module {module_name} not found")

    try:
        cur_obj = module
        for objname in function_name.split('.'):
            cur_obj = getattr(cur_obj, objname)
    except AttributeError:
        raise ModuleNotFoundError(f"Module attribute {module_name}.{objname} not found")
    return cur_obj


def initialize_automatic_logging():
    """
    This function initializes the automatic logging of Pytorch loss using patching.
    """

    params = [
        {
            'target': 'torch.Tensor.backward',
            'cb_before': _backward_callback
        },
        {
            'target': 'torch.tensor.Tensor.backward',
            'cb_before': _backward_callback
        }
    ]

    for p in params:
        try:
            Wrapper(experiment=None, **p).start()
        except Exception as e:
            _LOGGER.debug(f"Error while patching {p['target']}: {e}")
