import logging
import logging.config
from rich.console import ConsoleRenderable
from rich.logging import RichHandler
from rich.traceback import Traceback
import yaml
import importlib

_LOGGER = logging.getLogger(__name__)


class ConditionalRichHandler(RichHandler):
    """
    Class that uses 'show_level=True' only if the message level is WARNING or higher.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def handle(self, record):
        if record.levelno >= logging.WARNING:
            self.show_level = True
        else:
            self.show_level = False
        super().handle(record)

    def render(self, *, record: logging.LogRecord,
               traceback: Traceback | None,
               message_renderable: ConsoleRenderable) -> ConsoleRenderable:
        # if level is WARNING or higher, add the level column
        try:
            self._log_render.show_level = record.levelno >= logging.WARNING
            ret = super().render(record=record, traceback=traceback, message_renderable=message_renderable)
            self._log_render.show_level = False
        except Exception as e:
            _LOGGER.error(f"Error rendering log. {e}")
        return ret


def load_cmdline_logging_config():
    # Load the logging configuration file
    try:
        try:
            # try loading the developer's logging config
            with open('logging_dev.yaml', 'r') as f:
                config = yaml.safe_load(f)
        except:
            with importlib.resources.open_text('datamint', 'logging.yaml') as f:
                config = yaml.safe_load(f.read())

        logging.config.dictConfig(config)
    except Exception as e:
        print(f"Warning: Error loading logging configuration file: {e}")
        _LOGGER.exception(e)
        logging.basicConfig(level=logging.INFO)
