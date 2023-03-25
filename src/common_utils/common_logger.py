import logging
import sys


def get_logger(name: str):
    logger = logging.getLogger(name)
    formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    handler2 = logging.StreamHandler(sys.stdout)
    handler2.setFormatter(formatter)
    logger.addHandler(handler2)

    logger.setLevel(logging.DEBUG)

    return logger
