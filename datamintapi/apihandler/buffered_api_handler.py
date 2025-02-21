import threading
import queue
import json
from datamintapi.apihandler.base_api_handler import BaseAPIHandler
import logging
import atexit
import inspect


_LOGGER = logging.getLogger(__name__)


class BufferedAPIHandler:
    def __init__(self, api_handler: BaseAPIHandler,
                 buffer_size: int = 1,
                 flush_interval: int = 60,
                 save_path: str = "/tmp/buffered_logs.json"):
        self.api_handler = api_handler
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self.save_path = save_path
        self.log_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.check_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        atexit.register(self.stop)

    def _run(self):
        try:
            self._flush()
        except Exception as e:
            _LOGGER.error(f"Error in BufferedAPIHandler thread: {e}")

        while not self.stop_event.is_set():
            try:
                self.check_event.wait(timeout=self.flush_interval)
                self.check_event.clear()
                self._flush()
            except Exception as e:
                _LOGGER.error(f"Error in BufferedAPIHandler thread: {e}")

    def _flush(self):
        logs_to_upload = []
        while not self.log_queue.empty():  # and len(logs_to_upload) < self.buffer_size:
            logs_to_upload.append(self.log_queue.get())

        if len(logs_to_upload) > 0:
            try:
                for i in range(len(logs_to_upload)):
                    self._upload_log(logs_to_upload[i])
            except Exception as e:
                _LOGGER.error(f"Error uploading log: {e}")
                self._save_to_disk(logs_to_upload[i:])

    def _upload_log(self, log):
        method = getattr(self.api_handler, log['method'])
        method(*log['args'], **log['kwargs'])

    def _save_to_disk(self, logs):
        with open(self.save_path, 'a') as f:
            for log in logs:
                f.write(json.dumps(log) + '\n')

    def __getattr__(self, method_name: str):
        api_method = getattr(self.api_handler, method_name)
        if not callable(api_method):
            return api_method

        def method(*args, **kwargs):
            result = api_method(*args, **kwargs)
            if result is None:
                self.log_queue.put({'method': method_name, 'args': args, 'kwargs': kwargs})
                if self.log_queue.qsize() >= self.buffer_size:
                    self.check_event.set()
            return result

        if inspect.signature(api_method).return_annotation is None:
            _LOGGER.debug(f"Method {method_name} has no return annotation")
            return method
        else:
            return api_method

    def stop(self):
        _LOGGER.debug("Stopping BufferedAPIHandler")
        self.stop_event.set()
        self.check_event.set()
