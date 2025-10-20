from unittest.mock import patch
import importlib
from typing import Sequence, Any, Dict
import logging
from .experiment import Experiment
from torch.utils.data import DataLoader
import torch
import sys
import pandas as pd
import atexit
from collections import OrderedDict, defaultdict
import numpy as np

_LOGGER = logging.getLogger(__name__)

IS_INITIALIZED = False


def _is_iterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False


class Wrapper:
    class IteratorWrapper:
        def __init__(self, iterator,
                     cb_before_first_next,
                     cb_next_return,
                     cb_after_iter_return,
                     original_func,
                     cb_args,
                     cb_kwargs) -> None:
            self.iterator = iterator
            self.cb_args = cb_args
            self.cb_kwargs = cb_kwargs
            self.original_func = original_func
            self.cb_before_first_next = cb_before_first_next
            self.cb_next_return = cb_next_return
            self.cb_after_iter_return = cb_after_iter_return
            self.first_next = True

        def __iter__(self):
            self.iterator = iter(self.iterator)
            return self

        def __next__(self):
            if self.first_next:
                for cb in self.cb_before_first_next:
                    cb(self.original_func, self.cb_args, self.cb_kwargs, original_iter=self.iterator)
                self.first_next = False
            try:
                return_value = next(self.iterator)
                for cb in self.cb_next_return:
                    cb(self.original_func, self.cb_args, self.cb_kwargs, return_value)
                return return_value
            except StopIteration as e:
                for cb in self.cb_after_iter_return:
                    cb(self.original_func, self.cb_args, self.cb_kwargs, original_iter=self.iterator)
                self.first_next = True
                raise e

        def __getattr__(self, item):
            if item != '__iter__':
                return getattr(self.iterator, item)
            return self.__iter__

        def __len__(self):
            return len(self.iterator)

    def __init__(self,
                 target: str,
                 cb_before: Sequence[callable] | callable = None,
                 cb_after: Sequence[callable] | callable = None,
                 cb_before_first_next: Sequence[callable] | callable = None,
                 cb_next_return: Sequence[callable] | callable = None,
                 cb_after_iter_return: Sequence[callable] | callable = None,
                 ) -> None:
        self.cb_before = cb_before if cb_before is not None else []
        self.cb_after = cb_after if cb_after is not None else []
        self.cb_after_iter_return = cb_after_iter_return if cb_after_iter_return is not None else []
        self.cb_before_first_next = cb_before_first_next if cb_before_first_next is not None else []
        self.cb_next_return = cb_next_return if cb_next_return is not None else []
        if not _is_iterable(self.cb_before):
            self.cb_before = [self.cb_before]
        if not _is_iterable(self.cb_after):
            self.cb_after = [self.cb_after]
        if not _is_iterable(self.cb_after_iter_return):
            self.cb_after_iter_return = [self.cb_after_iter_return]
        if not _is_iterable(self.cb_before_first_next):
            self.cb_before_first_next = [self.cb_before_first_next]
        if not _is_iterable(self.cb_next_return):
            self.cb_next_return = [self.cb_next_return]
        self.target = target
        self._patch()

    def _patch(self):
        def _callback(*args, **kwargs):
            for cb in self.cb_before:
                cb(original, args, kwargs)

            try:
                return_value = original(*args, **kwargs)
                # if return_value is a generator, wrap it
                if len(self.cb_after_iter_return) > 0 and _is_iterable(return_value):
                    return_value = self._wrap_iterator(return_value,
                                                       original,
                                                       args, kwargs)

            except Exception as exception:
                # We are assuming the patched function does not return an exception.
                # return_value = exception
                raise exception

            for cb in self.cb_after:
                cb(original, args, kwargs, return_value)

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

    def _wrap_iterator(self, iterator, original_func, args, kwargs):
        return Wrapper.IteratorWrapper(iterator,
                                       cb_before_first_next=self.cb_before_first_next,
                                       cb_next_return=self.cb_next_return,
                                       cb_after_iter_return=self.cb_after_iter_return,
                                       original_func=original_func,
                                       cb_args=args,
                                       cb_kwargs=kwargs)


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


class PytorchPatcher:
    class DataLoaderInfo:
        def __init__(self, dataloader: DataLoader):
            self.dataloader = dataloader
            self.times_started_iter = 0  # This includes the current iteration
            self.is_iterating = False
            self.metrics = []
            self.iteration_idx = None
            self.predictions = defaultdict(list)  # TODO: save to disk
            self.cur_batch = None

            for obj in [dataloader, dataloader.batch_sampler]:
                if hasattr(obj, 'batch_size'):
                    self.batch_size = obj.batch_size
                    _LOGGER.debug(f"Found batch size {self.batch_size} for dataloader {dataloader}")
                    break
            else:
                self.batch_size = None
                _LOGGER.debug(f"Could not find batch size for dataloader {dataloader}")

        def __str__(self) -> str:
            return f"DataLoaderInfo: {self.dataloader}, number_of_times_iterated: {self.times_started_iter}, " \
                f"is_iterating: {self.is_iterating}"

        def append_metric(self, name: str, value: float, step, epoch):
            self.metrics.append([name, value, step, epoch])

    AUTO_LOSS_LOG_INTERVAL = 20

    def __init__(self) -> None:
        self.dataloaders_info: Dict[Any, PytorchPatcher.DataLoaderInfo] = OrderedDict()
        self.metrics_association = {}  # Associate metrics with dataloaders
        self.last_dataloader = None
        self.exit_with_error = False

    def _dataloader_created(self,
                            original_obj, func_args, func_kwargs,
                            return_value):

        dataloader = func_args[0]
        if dataloader in self.dataloaders_info:
            _LOGGER.warning("Dataloader already exists")
        _LOGGER.debug('Adding a new dataloader')
        self.dataloaders_info[dataloader] = PytorchPatcher.DataLoaderInfo(dataloader)

    def _inc_exp_step(self) -> int:
        exp = Experiment.get_singleton_experiment()
        if exp.cur_step is None:
            exp._set_step(0)
        else:
            exp._set_step(exp.cur_step + 1)

        return exp.cur_step

    def _backward_cb(self,
                     original_obj, func_args, func_kwargs):
        """
        This method is a wrapper for the backward method of the Pytorch Tensor class.
        """
        loss = func_args[0]
        cur_step = self._inc_exp_step()

        if cur_step % PytorchPatcher.AUTO_LOSS_LOG_INTERVAL == 0:
            self._log_metric('loss', loss.item())

    def clf_loss_computed(self,
                          original_obj, func_args, func_kwargs, return_value):
        loss = return_value.detach().cpu()
        # if is not a 0-d tensor, do not log
        if len(loss.shape) != 0:
            return
        loss = loss.item()
        loss_name = original_obj.__name__
        cur_step = self._inc_exp_step()

        dataloader = self.get_last_dataloader()
        if dataloader is not None and self.dataloaders_info[dataloader].is_iterating:
            self.metrics_association[func_args[0]] = dataloader

        if cur_step % PytorchPatcher.AUTO_LOSS_LOG_INTERVAL == 0:
            self._log_metric(loss_name, loss, dataloader=dataloader)

    def _classification_loss_computed(self, preds, targets):
        dataloader = self.get_last_dataloader()
        if dataloader is not None and self.dataloaders_info[dataloader].is_iterating:
            dinfo = self.dataloaders_info[dataloader]
            if not isinstance(dinfo.cur_batch, dict) or 'metainfo' not in dinfo.cur_batch:
                _LOGGER.debug(f"No metainfo in batch")
                return
            batch_metainfo = dinfo.cur_batch['metainfo']
            if 'id' not in batch_metainfo[0]:
                _LOGGER.debug("No id in batch metainfo")
                return
            resources_ids = [b['id'] for b in batch_metainfo]

            if len(resources_ids) != len(preds):
                _LOGGER.debug(f"Number of predictions ({len(preds)}) and targets ({len(targets)}) do not match")
                return

            dinfo.predictions['predictions'].extend(preds)
            dinfo.predictions['id'].extend(resources_ids)
            dinfo.predictions['frame_index'].extend([b['frame_index'] for b in batch_metainfo])
        else:
            _LOGGER.warning("No dataloader found")

    def bce_with_logits_computed(self,
                                 original_obj, func_args, func_kwargs, return_value):
        self.clf_loss_computed(original_obj, func_args, func_kwargs, return_value)
        preds = func_kwargs['input'] if 'input' in func_kwargs else func_args[0]
        targets = func_kwargs['target'] if 'target' in func_kwargs else func_args[1]
        preds = torch.nn.functional.sigmoid(preds).detach().cpu()
        targets = targets.detach().cpu()
        self._classification_loss_computed(preds, targets)

    def ce_computed(self,
                    original_obj, func_args, func_kwargs, return_value):
        self.clf_loss_computed(original_obj, func_args, func_kwargs, return_value)
        preds = func_kwargs['input'] if 'input' in func_kwargs else func_args[0]
        targets = func_kwargs['target'] if 'target' in func_kwargs else func_args[1]
        preds = preds.detach().cpu()
        targets = targets.detach().cpu()
        self._classification_loss_computed(preds, targets)

    def _dataloader_start_iterating_cb(self,
                                       original_obj, func_args, func_kwargs, original_iter):
        exp = Experiment.get_singleton_experiment()
        dataloader = func_args[0]  # self
        dataloader_info = self.dataloaders_info[dataloader]
        dataloader_info.is_iterating = True
        if dataloader_info.iteration_idx is None:
            dataloader_info.iteration_idx = self._get_dataloader_iteration_idx()+1
        exp._set_epoch(dataloader_info.times_started_iter)
        dataloader_info.times_started_iter += 1

        self.last_dataloader = dataloader

        _LOGGER.debug(f'Dataloader is iterating: {dataloader_info}')

    def _dataloader_next(self,
                         original_obj, func_args, func_kwargs, return_value):
        dataloader = func_args[0]
        dataloder_info = self.dataloaders_info[dataloader]
        dataloder_info.cur_batch = return_value

    def _dataloader_stop_iterating_cb(self,
                                      original_obj, func_args, func_kwargs, original_iter):
        dataloader = func_args[0]
        dinfo = self.dataloaders_info[dataloader]
        dinfo.is_iterating = False
        dinfo.cur_batch = None
        # find the dataloader that is still iterating # FIXME: For 3 dataloaders being iterating
        for dloader, dlinfo in self.dataloaders_info.items():
            if dlinfo.is_iterating:
                self.last_dataloader = dloader
                break
        else:
            _LOGGER.debug("No dataloader is iterating")

        _LOGGER.debug(f'Dataloader stopped iterating: {self.dataloaders_info[dataloader]}')

    def _log_metric(self, name, value,
                    dataloader=None,
                    **kwargs):
        exp = Experiment.get_singleton_experiment()
        if self.finish_callback not in exp.finish_callbacks:
            exp._add_finish_callback(self.finish_callback)

        if dataloader is None:
            dataloader = self.get_last_dataloader()
            dloader_info = self.dataloaders_info[dataloader]
            if not dloader_info.is_iterating:
                dataloader = None
        else:
            dloader_info = self.dataloaders_info[dataloader]

        if dataloader is not None:
            name = f"dataset{dloader_info.iteration_idx+1}/{name}"
            dloader_info.append_metric(name, value, exp.cur_step, exp.cur_epoch)

        _LOGGER.debug(f"Logging metric {name} with value {value}")
        exp.log_metric(name, value, **kwargs)

    def torchmetric_clf_computed(self,
                                 original_obj, func_args, func_kwargs, return_value):
        if isinstance(return_value, torch.Tensor):
            return_value = return_value.item()

        dataloader = self.metrics_association.get(func_args[0], None)

        self._log_metric(func_args[0].__class__.__name__,
                         value=return_value,
                         dataloader=dataloader)

    def torchmetric_clf_updated(self,
                                original_obj, func_args, func_kwargs):
        dataloader = self.get_last_dataloader()
        if dataloader is None or not self.dataloaders_info[dataloader].is_iterating:
            _LOGGER.debug("Dataloader not found or not iterating")
            return

        self.metrics_association[func_args[0]] = dataloader

    def _get_dataloader_iteration_idx(self) -> int:
        dataloader = self.get_last_dataloader()
        if dataloader is None:
            return -1
        return self.dataloaders_info[dataloader].iteration_idx

    def get_last_dataloader(self):
        return self.last_dataloader

    def _rename_metric(self, metric_name: str, phase: str) -> str:
        real_metric_name = metric_name.split('/', 1)[-1]
        if real_metric_name.startswith('Binary') or real_metric_name.startswith('Multiclass') or real_metric_name.startswith('Multilabel'):
            real_metric_name = real_metric_name.replace('Binary', '').replace(
                'Multiclass', '').replace('Multilabel', '')

        if real_metric_name == 'Recall':
            real_metric_name = 'Sensitivity'

        if real_metric_name == 'Precision':
            real_metric_name = 'Positive Predictive Value'

        if phase is not None:
            return f"{phase}/{real_metric_name}"

        return metric_name.split('/', 1)[0] + real_metric_name

    def finish_callback(self, exp: Experiment):
        # Get the last dataloader with 1 iteration, and assume it is the test dataloader
        dataloader = None
        phase = None
        for dloader, dlinfo in reversed(self.dataloaders_info.items()):
            if dlinfo.times_started_iter == 1:
                dataloader = dloader
                phase = 'test'
                break
        else:
            _LOGGER.debug('No dataloader with 1 iteration found')
        if dataloader is None:
            dataloader = self.get_last_dataloader()
            if len(self.dataloaders_info) > 1:
                phase = 'test'
            else:
                _LOGGER.warning("No test dataloader found")
        if dataloader is None:
            _LOGGER.warning("No dataloader to log found")
            return

        dlinfo = self.dataloaders_info[dataloader]

        # log predictions
        if len(dlinfo.predictions) > 0:
            if hasattr(dataloader.dataset, 'labels_set'):
                exp.log_classification_predictions(predictions_conf=np.array(dlinfo.predictions['predictions']),
                                                   label_names=dataloader.dataset.labels_set,
                                                   resource_ids=dlinfo.predictions['id'],
                                                   dataset_split=phase,
                                                   frame_idxs=dlinfo.predictions['frame_index'])

        dlinfo_metrics = pd.DataFrame(dlinfo.metrics,
                                      columns=['name', 'value', 'step', 'epoch'])
        summary = {'metrics': {}}
        # only use value from the last epoch
        dlinfo_metrics = dlinfo_metrics[dlinfo_metrics['epoch'] == dlinfo_metrics['epoch'].max()]

        for metric_name, value in dlinfo_metrics.groupby('name')['value'].mean().items():
            metric_name = self._rename_metric(metric_name, phase)
            summary['metrics'][metric_name] = value

        exp.add_to_summary(summary)

    def module_constructed_cb(self,
                              original_obj, func_args, func_kwargs, value):
        exp = Experiment.get_singleton_experiment()
        if exp is not None and exp.model is None:
            model = func_args[0]

            # check that is not a torchmetrics model
            if model.__module__.startswith('torchmetrics.'):
                return
            # Not a loss function
            if model.__module__.startswith('torch.nn.modules.loss'):
                return
            # Not a torchvision transform
            if model.__module__.startswith('torchvision.transforms'):
                return
            # Not a optimizer
            if model.__module__.startswith('torch.optim'):
                return

            exp.set_model(model)
            _LOGGER.debug(f'Found user model {model.__class__.__name__}')

    def custom_excepthook(self, exc_type, exc_value, traceback):
        ORIGINAL_EXCEPTHOOK
        self.exit_with_error = True
        # Call the original exception hook
        ORIGINAL_EXCEPTHOOK(exc_type, exc_value, traceback)

    def at_exit_cb(self):
        if self.exit_with_error:
            return
        exp = Experiment.get_singleton_experiment()
        if exp is not None and (exp.cur_step is not None or exp.cur_epoch is not None):
            exp.finish()


def initialize_automatic_logging(enable_rich_logging: bool = True):
    """
    This function initializes the automatic logging of Pytorch loss using patching.
    """
    from rich.logging import RichHandler
    global IS_INITIALIZED, ORIGINAL_EXCEPTHOOK

    if IS_INITIALIZED == True:
        return
    IS_INITIALIZED = True

    # check if RichHandler is already in the handlers
    if enable_rich_logging and not any(isinstance(h, RichHandler) for h in logging.getLogger().handlers):
        logging.getLogger().handlers.append(RichHandler())  # set rich logging handler for the root logger
    # logging.getLogger("datamint").setLevel(logging.INFO)

    pytorch_patcher = PytorchPatcher()

    torchmetrics_clfs_base_metrics = ['Recall', 'Precision', 'AveragePrecision',
                                      'F1Score', 'Accuracy', 'AUROC',
                                      'CohenKappa']

    torchmetrics_clf_metrics = [f'Multiclass{m}' for m in torchmetrics_clfs_base_metrics]
    torchmetrics_clf_metrics += [f'Multilabel{m}' for m in torchmetrics_clfs_base_metrics]
    torchmetrics_clf_metrics += [f'Binary{m}' for m in torchmetrics_clfs_base_metrics]
    torchmetrics_clf_metrics = [f'torchmetrics.classification.{m}' for m in torchmetrics_clf_metrics]
    torchmetrics_detseg_metrics = ['torchmetrics.segmentation.GeneralizedDiceScore',
                                   'torchmetrics.detection.iou.IntersectionOverUnion',
                                   'torchmetrics.detection.giou.GeneralizedIntersectionOverUnion']

    torchmetrics_metrics = torchmetrics_clf_metrics + torchmetrics_detseg_metrics

    params = [
        {
            'target': ['torch.Tensor.backward', 'torch.tensor.Tensor.backward'],
            'cb_before': pytorch_patcher._backward_cb
        },
        {
            'target': 'torch.utils.data.DataLoader.__iter__',
            'cb_before_first_next': pytorch_patcher._dataloader_start_iterating_cb,
            'cb_after_iter_return': pytorch_patcher._dataloader_stop_iterating_cb,
            'cb_next_return': pytorch_patcher._dataloader_next,
        },
        {
            'target': 'torch.utils.data.DataLoader.__init__',
            'cb_after': pytorch_patcher._dataloader_created
        },
        {
            'target': 'torch.nn.functional.nll_loss',
            'cb_after': pytorch_patcher.clf_loss_computed
        },
        {
            'target': [f'{m}.compute' for m in torchmetrics_metrics],
            'cb_after': pytorch_patcher.torchmetric_clf_computed
        },
        {
            'target': [f'{m}.update' for m in torchmetrics_metrics],
            'cb_before': pytorch_patcher.torchmetric_clf_updated
        },
        {
            'target': 'torch.nn.modules.module.Module.__init__',
            'cb_after': pytorch_patcher.module_constructed_cb
        },
        {
            'target': 'torch.nn.functional.binary_cross_entropy_with_logits',
            'cb_after': pytorch_patcher.bce_with_logits_computed
        },
        {
            'target': ['torch.nn.functional.binary_cross_entropy', 'torch.nn.functional.cross_entropy'],
            'cb_after': pytorch_patcher.ce_computed
        }
    ]

    # explode the list of targets into individual targets
    new_params = []
    for p in params:
        if isinstance(p['target'], list):
            for t in p['target']:
                new_params.append({**p, 'target': t})
        else:
            new_params.append(p)
    params = new_params

    for p in params:
        try:
            Wrapper(**p).start()
        except Exception as e:
            _LOGGER.debug(f"Error while patching {p['target']}: {e}")

    try:
        # Set the custom exception hook
        ORIGINAL_EXCEPTHOOK = sys.excepthook
        sys.excepthook = pytorch_patcher.custom_excepthook
        atexit.register(pytorch_patcher.at_exit_cb)
    except Exception:
        _LOGGER.warning("Failed to use atexit.register")
