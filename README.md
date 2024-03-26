# Python library : torch_hog

The torch_hog library provides implementation for calculating the Histogram Of Oriented Gradients feature descriptors.


# References :

- [KNN] "Histograms of Oriented Gradients for Human Detection" Navneet Dalal and Bill Triggs - https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf



# Install library



```bash
%%bash
if !python -c "import torch_hog" 2>/dev/null; then
    pip install https://github.com/Simon-Bertrand/HOG-PyTorch/archive/main.zip
fi
```

# Import library



```python
import torch_hog
```

# Load data and and compute ground truth


```python
!pip install -q scikit-image
```

    
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m A new release of pip available: [0m[31;49m22.2.1[0m[39;49m -> [0m[32;49m24.0[0m
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m To update, run: [0m[32;49mpip install --upgrade pip[0m



```python
from skimage import feature
from skimage import data
import numpy as np
import torch
import torch_hog
image = data.astronaut().astype(np.float32)
im = torch.Tensor(image).moveaxis(-1, 0).unsqueeze(0)

featuresGt, hogImFt = feature.hog(
    image,
    orientations=9,
    pixels_per_cell=(16, 16),
    cells_per_block=(2, 2),
    block_norm="L2",
    visualize=True,
    channel_axis=-1,
    feature_vector=False,
)
hog = torch_hog.HOG(
    cellSize=16,
    blockSize=2,
    kernel="finite",
    normalization="L2",
    accumulate="simple",
    channelWise=False,
)
hogIm = hog.visualize(im, orthogonal=True)
```

# Visualization comparison with skimage


```python
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(16, 16))

axes[0].imshow(image / 255)
axes[0].set_title("SkImage")
axes[0].imshow(
    hogImFt, cmap="gray", alpha=0.75
)  
axes[1].imshow(image / 255)
axes[1].set_title("torch_hog")
axes[1].imshow(
    hogIm[0,0], cmap="gray", alpha=0.75
)  
```




    <matplotlib.image.AxesImage at 0x7fc1884eaa40>




    
![png](figs/README_10_1.png)
    


# Testing with skimage


```python
features = hog(im)
assert (features[0,0]-torch.Tensor(featuresGt)).abs().max()<1e-6
(features[0,0]-torch.Tensor(featuresGt)).mean()
```




    tensor(3.1590e-10)



# Visualize on max norm channel


```python
hog = torch_hog.HOG(
    cellSize=16,
    blockSize=2,
    kernel="finite",
    normalization="L2",
    accumulate="simple",
    channelWise=False,
)
hogIm = hog.plotVisualize(im)
```


    
![png](figs/README_14_0.png)
    


# Visualize channel wise


```python
hog = torch_hog.HOG(
    cellSize=16,
    blockSize=2,
    kernel="finite",
    normalization="L2",
    accumulate="simple",
    channelWise=True,
)
hogIm = hog.plotVisualize(im)
```


    
![png](figs/README_16_0.png)
    


# Bilinear accumulating


```python
hog = torch_hog.HOG(
    cellSize=16,
    blockSize=2,
    kernel="finite",
    normalization="L2",
    accumulate="bilinear",
    channelWise=False,
)
hogIm = hog.plotVisualize(im)
```


    
![png](figs/README_18_0.png)
    



```python

```
