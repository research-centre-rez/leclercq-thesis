#TODO:

- check automatic thresholding algorithms if needed
- how to measure the orientation of the crack wrt to the rotating cylinder
- how to deal with cracks going invisible? -> store them even if we can't seem em??
- Fitting ellipse (Ferret) method for quantifying cracks?
- Check out skeletonisation stuff, might have some useful information
- OpenCV trackers
- YOLO model??
- establish some sort of pipeline for how this is going to go

# Video analysis

Videos are just a bunch of frames that go after each other, therefore I could probably just extract *some frames* instead of all of them.

It might still be a good idea to check for a way of rotating the image around (instead of the cylinder rotating the whole image rotates such that the cylinder appears "static")
# Image analysis

Storing the video / images as different representations.

Fully automated process might not be perfect, maybe some input from the user might be of some help?

There is something called opencv trackers?? Might be worth looking into

## Binarisation

Converting of an image into a black and white image instead of working with grayscale might be a good idea. This can either be done manually or by applying automatic thresholding algorithms. It might be possible to identify branches of the skeleton of the cracks.

Binarisation allows for separating the *inidividual* cracks, instead of having one big crack that kinda "behaves" like a junction, you can separate them up into smaller cracks. 

> [!THOUGHT]
> Maybe it would be worth to take all of the smaller cracks and store *those* into a feature vector? 

## Geometrical attributes

Length can be calculated as the sum of pixels of the skeletonised object, multiplied by a factor derived from the orientation (which ranges from 1 to sqrt(2)) this is then used to convert pixels to "linear units". You can visualise cracks by colouring them differently, it looks kinda cool :0

Cracks usually show up as winding lines, which is why there is a need of geometrical approximation when calculating their overall orientation. This is done by subdividing each object in a number of straight segments defined by the intersection of the crack. 

The dataset might contain linear elements that are not cracks (lamellae?, scratches). This might need to be filtered out.
