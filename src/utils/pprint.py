import logging

"""
This file contains various pretty print methods for different data types
"""
def pprint_argparse(args, logger=None):
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info('Running with the following parameters:')
    for arg in vars(args):
        logger.info('  %s: %s', arg, getattr(args, arg))

def pprint_dict(dict, desc, logger=None):
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info(desc)
    for key, val in dict.items():
        logger.info(' %s: %s', key, val)
