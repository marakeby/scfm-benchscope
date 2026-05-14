import logging
import datetime
import os
import sys

def set_logging(log_dir):

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    print (log_dir)
    filename = os.path.join(log_dir,  'log.log')
    logging.basicConfig(filename = filename,
                        filemode='w',
                        format='%(asctime)s - {%(filename)s:%(lineno)d} - %(message)s',
                        datefmt='%m/%d %I:%M',
                        level=logging.INFO, encoding="utf8") # or logging.DEBUG
    logging.getLogger('sc_ml').addHandler(logging.StreamHandler(sys.stdout))
    logging.info('setting logs')
    return logging.getLogger('sc_ml')

def get_logger():
    return logging.getLogger('sc_ml')

