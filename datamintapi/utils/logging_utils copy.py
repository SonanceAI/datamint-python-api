from logging import Logger
import logging


class LevelLoggerSwitch:
    """
    A class that switches between two loggers based on the level of the log message.
    """

    def __init__(self,
                 info_logger: Logger | str,
                 warning_logger: Logger | str):
        if isinstance(info_logger, str):
            self.info_logger = logging.getLogger(info_logger)
        else:
            self.info_logger = info_logger
        if isinstance(warning_logger, str):
            self.warning_logger = logging.getLogger(warning_logger)
        else:
            self.warning_logger = warning_logger

    def info(self, msg, *args, **kwargs):
        self.info_logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self.warning_logger.warning(msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        self.info_logger.debug(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.warning_logger.error(msg, *args, **kwargs)

    def exception(self, msg, *args, **kwargs):
        self.warning_logger.exception(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        self.warning_logger.critical(msg, *args, **kwargs)

    def log(self, level, msg, *args, **kwargs):
        if level >= logging.WARNING:
            self.warning_logger.log(level, msg, *args, **kwargs)
        else:
            self.info_logger.log(level, msg, *args, **kwargs)

    def setLevel(self, level):
        self.info_logger.setLevel(level)
        self.warning_logger.setLevel(level)

    def isEnabledFor(self, level):
        return self.info_logger.isEnabledFor(level) or self.warning_logger.isEnabledFor(level)

    def addHandler(self, handler):
        self.info_logger.addHandler(handler)
        self.warning_logger.addHandler(handler)

    def removeHandler(self, handler):
        self.info_logger.removeHandler(handler)
        self.warning_logger.removeHandler(handler)
