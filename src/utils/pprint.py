import logging

"""
This file contains various pretty print methods for different data types
"""
logger = logging.getLogger(__name__)
def pprint_argparse(args):
    logger.info('Running with the following parameters:')
    for arg in vars(args):
        logger.info('  %s: %s', arg, getattr(args, arg))

def pprint_dict(dict_in, desc):
    logger.info(desc)
    for key, val in dict_in.items():
        if isinstance(val, dict):
            pprint_dict(val, key.upper())
        else:
            logger.info(' %s: %s', key, val)
