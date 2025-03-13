from tqdm import tqdm

def tqdm_generator():
    '''
    Simple tqdm generator that allows the use of tqdm with a while loop.
    '''
    while True:
        yield
