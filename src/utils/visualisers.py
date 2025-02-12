import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

# Leclercq's util functions

def show_overlay(saving_dest:str, img:cv.typing.MatLike, mask:cv.typing.MatLike, alpha=0.5) -> None:
    '''
    Saves an overlay of an image and its mask to `saving_dest`. The mask is shown in a green colour.
    Args:
        saving_dest: where to save the output of the image
        img: opencv image
        mask: single channel mask
        alpha: how transparent should the mask be, if 1 then there is no transparency
    '''
    colour_mask = cv.merge([mask * 0, mask, mask * 0])
    overlaid    = cv.addWeighted(img, 1, colour_mask, alpha, 0)

    cv.imwrite(saving_dest, overlaid)

def imshow(title= None, **images) -> None:
    '''
    Displays images in one row.
    Args:
        title: What the title of the plot should be
        **images: the name of the variable determines its title in the graph. 
            - `image` variable is reserved for torch tensors, channels are permuted
            - else it prints an image
    '''

    n = len(images)

    cols = min(n, 3)
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*3), constrained_layout=True)

    if rows == 1:
        axes = np.array(axes.reshape(1,-1))
    if cols == 1:
        axes = np.array(axes.reshape(1,-1))

    axes = axes.flatten()

    for ax, (name, image) in zip(axes, images.items()):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(' '.join(name.split('_')).title())

        if name == 'image':
            ax.imshow(image.permute(1,2,0))
        else:
            ax.imshow(image)

    for ax in axes[len(images):]:
        ax.axis('off')

    plt.savefig('result' if title is None else title, bbox_inches='tight', dpi=400)

    plt.show()
    plt.pause(0.1)
