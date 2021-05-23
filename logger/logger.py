import logging

import sys

APP_LOGGER_NAME = 'RealTimeFaceDetection'

def setup_logger(logger_name = APP_LOGGER_NAME, file_name = None):

    logger = logging.getLogger(logger_name)

    # setting log level as debug
    logger.setLevel(logging.DEBUG)

    # setting log format
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # stream handler to write log messages
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)

    # adding stream handler
    logger.handlers.clear()
    logger.addHandler(sh)

    # creating file to store the logs
    if file_name:

        fh = logging.FileHandler(file_name)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger

def get_logger(module_name):
    return logging.getLogger(APP_LOGGER_NAME).getChild(module_name)