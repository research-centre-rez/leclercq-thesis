import logging

def pprint_argparse(args, logger=None):
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info('Running with the following parameters:')
    for arg in vars(args):
        logger.info(f'  {arg}: {getattr(args,arg)}')

