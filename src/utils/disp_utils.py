import numpy as np

def extract_medians(displacement:np.ndarray):
    '''
    Extracts medians from displacement. Creates one median displacement instead of NxM displacement.
    '''
    medians = []
    for i in range(displacement.shape[-1]):
        x_med = np.median(displacement[0, :, :, i])
        y_med = np.median(displacement[1, :, :, i])
        medians.append([x_med, y_med])

    meds = np.array(medians)

    return meds

def extract_means(displacement:np.ndarray):
    '''
    Extracts means from displacement. Creates one mean displacement instead of NxM displacement.
    '''
    means = []
    for i in range(displacement.shape[-1]):
        x_mean = displacement[0, :, :, i].mean()
        y_mean = displacement[1, :, :, i].mean()
        means.append([x_mean, y_mean])

    means = np.array(means)

    return means

