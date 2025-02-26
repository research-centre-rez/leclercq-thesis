from circle_fit import taubinSVD, plot_data_circle
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from skimage.measure import EllipseModel
import sys


#31.14389212,  5.11694245 +
#14.08033127, 19.36611469
#------------------------
#45.22422339, 24.48305714

#62.28778424 10.2338849
#76.36811551 29.59999959

#66.73625786721902

def plot_ellipse(disp, xc,yc,a,b,theta):
    fig, ax = plt.subplots()
    ax.plot(disp[:,0], disp[:,1], 'bo', label='Data')
    # Generate fitted ellipse outline
    fit_t = np.linspace(0, 2*np.pi, 300)
    x_fit = xc + a * np.cos(fit_t) * np.cos(theta) - b * np.sin(fit_t) * np.sin(theta)
    y_fit = yc + a * np.cos(fit_t) * np.sin(theta) + b * np.sin(fit_t) * np.cos(theta)
    ax.plot(x_fit, y_fit, 'r-', label='Fitted ellipse')
    ax.plot(xc,yc, 'o', label='Center of ellipse')

    ax.set_aspect('equal', 'datalim')
    ax.legend()
    plt.show()

def find_ellipse(disp:np.ndarray):

    model = EllipseModel()
    success = model.estimate(disp)

    if not success:
        print('failed to find an ellipse')
        sys.exit()

    xc, yc, _, _, _ = model.params
    plot_ellipse(disp, *model.params)
    print(f'Center of ellipse: {(xc,yc)}')
    return xc,yc



def find_circle_for_center(displacement:np.ndarray, mesh_nodes:np.ndarray):
    '''
    Args:
        displacement: should be of shape [2,2,i,j,n]
        mesh_nodes: should be of shape [n, 2]
    '''

    print(displacement.shape)
    print(displacement[3:])

    xc, yc, r, sigma = taubinSVD(displacement)
    plot_data_circle(displacement, xc, yc, r)


if __name__ == "__main__":
    data = np.load('3A-part0_rotated_displacement.npz')
    mesh = data['mesh_nodes']
    disp = data['displacement']
    disp = disp.squeeze()

    disp = disp[:, 2, 2, :]
    center_mesh = mesh[12]

    disp = disp.transpose((1,0))
    disp[:, 1] = -disp[:, 1]
    print(disp)

    disp = disp + center_mesh

    xc, yc, r, sigma = taubinSVD(disp)
    print(xc,yc)
    print(r)
    print(sigma)

    offset = (xc, yc) - disp[0]
    print(offset)
    print(offset * 2)
    plot_data_circle(disp, xc, yc, r)
