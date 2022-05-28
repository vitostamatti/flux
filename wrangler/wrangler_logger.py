from loguru import logger as gurulogger
import sys


class WranglerLogger():
    _default_format = "{time:YYYY-MM-DD at HH:mm:ss} | {level} | {module} | {message}"
    def __init__(self, logger) -> None:
        self._logger = logger
        self._logger.remove(0)
        self._console_handler = None
        self._file_handler = None
        

    def enable(self, level='DEBUG'):
        if not self._console_handler:
            self._console_handler = self._logger.add(sys.stderr, format=self._default_format, level = level)
        else:
            self._logger.remove(self._console_handler)
            self._console_handler = self._logger.add(sys.stderr, format=self._default_format, level = level)
        self._logger.enable("wrangler")

    def disable(self):
        if self._console_handler:
            self._logger.remove(self._console_handler)
            self._console_handler = None
        else:
            pass
            # raise Exception("Console handler has not been defined")

    def enable_file(self, filename, level='DEBUG'):
        if not self._file_handler:
            self._file_handler = self._logger.add(filename, format=self._default_format, level=level)
        else:
            self._logger.remove(self._file_handler)
            self._file_handler = self._logger.add(filename, format=self._default_format, level=level)

    def disable_file(self):
        if self._file_handler:
            self._logger.remove(self._file_handler)
        else:
            pass
            # raise Exception("File handler has not been defined")
    
logger = WranglerLogger(gurulogger)




