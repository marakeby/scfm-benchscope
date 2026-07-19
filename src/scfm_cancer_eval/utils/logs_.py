import logging
import datetime
import os
import sys

def set_logging(log_dir):

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    print (log_dir)
    filename = os.path.join(log_dir,  'log.log')

    logger = logging.getLogger('sc_ml')
    logger.setLevel(logging.INFO)
    # Avoid stacking handlers across grid-search trials (duplicate stdout / cross-talk).
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt='%(asctime)s - {%(filename)s:%(lineno)d} - %(message)s',
        datefmt='%m/%d %I:%M',
    )
    file_handler = logging.FileHandler(filename, mode='w', encoding="utf8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    # Keep root configured for libraries that log via root, but don't duplicate our handlers.
    logging.basicConfig(
        format='%(asctime)s - {%(filename)s:%(lineno)d} - %(message)s',
        datefmt='%m/%d %I:%M',
        level=logging.INFO,
        encoding="utf8",
        force=True,
    )
    logging.info('setting logs')
    logger.info('setting logs')
    return logger

def get_logger():
    return logging.getLogger('sc_ml')
