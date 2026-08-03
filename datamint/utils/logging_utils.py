from rich.theme import Theme
from logging import Logger, DEBUG, INFO, WARNING, ERROR, CRITICAL
from rich.console import Console
import platform
import os
import logging
import logging.config
from rich.console import ConsoleRenderable
from rich.logging import RichHandler
from rich.traceback import Traceback
import yaml
import importlib
import datetime

_LOGGER = logging.getLogger(__name__)

_FILE_HANDLER_MARKER = '_datamint_file_handler'
_FALSY_ENV_VALUES = {'0', 'false', 'no'}


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

    # dictConfig replaces the handlers of the loggers it configures, so the file
    # handler must be (re)attached after it runs.
    setup_file_logging_if_enabled()


def _file_logging_enabled() -> bool:
    value = os.environ.get('DATAMINT_LOG_FILE')
    if value is None:
        return True
    return value.strip().lower() not in _FALSY_ENV_VALUES


def _has_file_handler(logger: logging.Logger) -> bool:
    return any(getattr(handler, _FILE_HANDLER_MARKER, False) for handler in logger.handlers)


def setup_file_logging_if_enabled():
    """
    Attaches a per-run file handler to the 'datamint' and 'user_logger' loggers,
    writing to ./.log/, unless DATAMINT_LOG_FILE is explicitly disabled.
    """
    try:
        if not _file_logging_enabled():
            return

        datamint_logger = logging.getLogger('datamint')
        user_logger = logging.getLogger('user_logger')

        if _has_file_handler(datamint_logger) or _has_file_handler(user_logger):
            return

        log_dir = os.path.join(os.getcwd(), '.log')
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"datamint_{timestamp}_{os.getpid()}.log"
        filepath = os.path.join(log_dir, filename)

        handler = logging.FileHandler(filepath)
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        setattr(handler, _FILE_HANDLER_MARKER, True)

        for logger in (datamint_logger, user_logger):
            logger.addHandler(handler)
            if logger.getEffectiveLevel() > logging.INFO:
                logger.setLevel(logging.INFO)
    except Exception as e:
        print(f"Warning: Could not set up file logging: {e}")


LEVELS_MAPPING = {
    DEBUG: None,
    INFO: None,
    WARNING: "warning",
    ERROR: "error",
    CRITICAL: "error"
}


def _create_console_theme() -> Theme:
    """Create a custom Rich theme optimized for cross-platform terminals."""
    # Detect if we're likely on PowerShell (Windows + PowerShell)
    is_powershell = (
        platform.system() == "Windows" and
        os.environ.get("PSModulePath") is not None
    )

    if is_powershell:
        # PowerShell blue background - use high contrast colors
        return Theme({
            "warning": "bright_yellow",
            "error": "bright_red on white",
            "success": "bright_green",
            "key": "bright_cyan",
            "accent": "bright_cyan",
            "title": "bold"
        })
    else:
        # Linux/Unix terminals - standard colors
        return Theme({
            "warning": "yellow",
            "error": "red",
            "success": "green",
            "key": "cyan",
            "accent": "bright_blue",
            "title": "bold"
        })


class ConsoleWrapperHandler(ConditionalRichHandler):
    """
    A logging handler that uses a rich.console.Console to print log messages.
    """
    def __init__(self, *args, console: Console | None = None, **kwargs):
        """
        Initializes the ConsoleWrapperHandler.

        Args:
            console (Console | None): A rich Console instance. If None, a new one is created.
        """
        super().__init__(*args, **kwargs)
        if console is None:
            console = Console(theme=_create_console_theme())
        self.console = console

    def emit(self, record: logging.LogRecord) -> None:
        """
        Emits a log record.

        Args:
            record (logging.LogRecord): The log record to emit.
        """
        try:
            msg = self.format(record)
            style = LEVELS_MAPPING.get(record.levelno)
            self.console.print(msg, style=style)
        except Exception:
            self.handleError(record)
