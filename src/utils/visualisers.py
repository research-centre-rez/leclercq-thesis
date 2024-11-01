import matplotlib.pyplot as plt

def imshow(title= None, **images):
    """
    Displays images in one row.
    Args:
        title: What the title of the plot should be
        **images: the name of the variable determines its title in the graph. 
            - `image` variable is reserved for torch tensors, channels are permuted
            - else it prints an image
    """

    n = len(images)
    plt.figure(figsize=(n*4,5))

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
