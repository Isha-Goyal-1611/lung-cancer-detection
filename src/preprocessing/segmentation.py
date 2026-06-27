import numpy as np
import pydicom
import matplotlib.pyplot as plt
from skimage import morphology, measure
ds = pydicom.dcmread('data/raw/sample.dcm', force=True)
pixel_array = ds.pixel_array
hu_image = pixel_array * ds.RescaleSlope + ds.RescaleIntercept

binary_mask = hu_image < -400

cleaned_mask = morphology.binary_erosion(binary_mask)
cleaned_mask = morphology.binary_dilation(cleaned_mask)

plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.imshow(hu_image, cmap='gray')
plt.title('HU Image')
plt.colorbar()
plt.subplot(1,3,2)
plt.imshow(binary_mask, cmap='gray')
plt.title('Binary Mask')
plt.colorbar()
plt.subplot(1,3,3)
plt.imshow(cleaned_mask, cmap='gray')
plt.title('Cleaned Mask')
plt.colorbar()
plt.savefig('outputs/day5_segmentation.png')
plt.show()

labeled_mask = measure.label(cleaned_mask)

regions = measure.regionprops(labeled_mask)



sorted_regions = sorted(regions, key=lambda x: x.area, reverse=True)
top_2= sorted_regions[:2]

lung_mask = np.zeros_like(labeled_mask)
for region in top_2:
    lung_mask[labeled_mask == region.label] = 1

plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.imshow(cleaned_mask, cmap='gray')
plt.title('Cleaned Mask')
plt.colorbar()
plt.subplot(1,3,2)
plt.imshow(labeled_mask, cmap='nipy_spectral')
plt.title('Labeled Mask')
plt.colorbar()
plt.subplot(1,3,3)
plt.imshow(lung_mask, cmap='gray')
plt.title('Lung Mask')
plt.colorbar()
plt.savefig('outputs/day6_lung_mask.png')
plt.show()



