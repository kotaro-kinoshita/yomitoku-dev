from logging import DEBUG, Formatter, StreamHandler, getLogger
import warnings


def set_logger(name, level="INFO"):
    logger = getLogger(name)
    logger.setLevel(DEBUG)
    handler = StreamHandler()
    handler.setLevel(level)
    format = Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(format)
    logger.addHandler(handler)

    warnings.filterwarnings("ignore")
    return logger
