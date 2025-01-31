import matplotlib.pyplot as plt
import cv2 as cv

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
    plt.figure(figsize=(n*4,3))

    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.title(" ".join(name.split("_")).title())
        if name=='image':
            plt.imshow(image.permute(1, 2, 0))
        else:
            plt.imshow(image)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('result' if title is None else title)

    plt.show()
    plt.pause(0.1)
