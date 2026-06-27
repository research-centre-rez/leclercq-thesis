---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.3
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Purpose of this notebook is:
1) to visualize CT scans from Vavřík. To solve their loading and visualization.
2) to see how complex is registration of scans before and after aging.

```python
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
```

```python
ROOT = "/Users/gimli/cvr/data/beton/2026-06-CT/"
before_pth = os.path.join(ROOT, "before", "1A_1512x1512x2808_22d84.raw")
after_pth = os.path.join(ROOT, "after", "1A_1512x1512x2808_22d86.raw")
```

```python
before = np.fromfile(before_pth, dtype=np.uint16).reshape(
    (2808, 1512, 1512)
)
after = np.fromfile(after_pth, dtype=np.uint16).reshape(
    (2808, 1512, 1512)
)
```

```python
# Source - https://stackoverflow.com/a/44874588
# Posted by alkasm, modified by community. See post 'Timeline' for change history
# Retrieved 2026-05-20, License - CC BY-SA 4.0

def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

```

```python
b_mask = create_circular_mask(1512, 1512, (760, 748), 496)
```

```python
plt.figure(figsize=(15, 15))
ax = plt.subplot(121)
ax.imshow(before[1050, :, :] * b_mask, cmap="gray")
ax.scatter([690, 318, 925], [1160, 718, 298], s=100, c='red', marker='+')
#Drawing_colored_circle = plt.Circle(( 760 , 748 ), 496, alpha=0.5, color='red')
#ax.add_artist( Drawing_colored_circle)
ax.set_xlim(np.where(np.sum(b_mask, axis=0) > 0)[0][0], np.where(np.sum(b_mask, axis=0) > 0)[0][-1])
ax.set_ylim(np.where(np.sum(b_mask, axis=1) > 0)[0][0], np.where(np.sum(b_mask, axis=1) > 0)[0][-1])
#plt.ylim(700, 800)
ax = plt.subplot(122)
ax.imshow(after[1062, :, :] * b_mask, cmap="gray")
ax.scatter([721, 309, 870], [1178, 778, 298], s=100, c='red', marker='+', alpha=0.5)
#Drawing_colored_circle = plt.Circle(( 760 , 748 ), 496, alpha=0.5, color='red')
#ax.add_artist( Drawing_colored_circle)
ax.set_xlim(np.where(np.sum(b_mask, axis=0) > 0)[0][0], np.where(np.sum(b_mask, axis=0) > 0)[0][-1])
ax.set_ylim(np.where(np.sum(b_mask, axis=1) > 0)[0][0], np.where(np.sum(b_mask, axis=1) > 0)[0][-1])

plt.show()
```

```python
def minmaxnorm(img):
    return (img - np.min(img))/(np.max(img) - np.min(img) + 1e-10)
```

```python
a = (after[1062, :, :] - np.min(after[1062, :, :]))/(np.max(after[1062, :, :])-np.min(after[1062, :, :]))
b = (before[1050, :, :] - np.min(before[1062, :, :]))/(np.max(before[1062, :, :])-np.min(before[1062, :, :]))
```

```python
T = cv2.getAffineTransform(np.array([[690, 318, 925], [1160, 718, 298]]).T.astype(np.float32),
                       np.array([[721, 309, 870], [1178, 778, 298]]).T.astype(np.float32))
```

```python
plt.figure(figsize=(15, 15))
plt.imshow(np.stack([minmaxnorm(a),
                     minmaxnorm(cv2.warpAffine(b, T, (1512, 1512))),
                     (minmaxnorm(a) + minmaxnorm(cv2.warpAffine(b, T, (1512, 1512)))) / 2], axis=2), cmap="gray")
plt.xlim(np.where(np.sum(b_mask, axis=0) > 0)[0][0], np.where(np.sum(b_mask, axis=0) > 0)[0][-1])
plt.ylim(np.where(np.sum(b_mask, axis=1) > 0)[0][0], np.where(np.sum(b_mask, axis=1) > 0)[0][-1])
plt.show()
```

```python
vb = [np.var(layer) for layer in before]
va = [np.var(layer) for layer in after]
```

```python
plt.figure(figsize=(15, 5))
plt.plot(vb)
plt.plot(va)
plt.xlim(1200, 1500)
plt.axvline(1256)
plt.axvline(1268)
plt.axvline(1397)
plt.axvline(1409)
plt.show()
```

### Make video from raw 8-bit grayscale

```python
size = 1512, 1512
fps = 25
fourcc = cv2.VideoWriter_fourcc(*"Y800")  # raw 8-bit grayscale
out = cv2.VideoWriter(
    os.path.join(ROOT, "out_gray_raw.avi"),
    fourcc,
    fps,
    size,
    isColor=False,
)
for f in data:
    frame = (f // 256).astype(np.uint8)
    assert frame.dtype == np.uint8, frame.dtype
    assert frame.ndim == 2, frame.shape
    assert frame.shape == size, frame.shape
    assert frame.flags["C_CONTIGUOUS"]
    out.write(frame)
out.release()
```

```python

```
