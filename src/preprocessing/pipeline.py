import numpy as np
import pydicom
from skimage import morphology, measure
import matplotlib.pyplot as plt
import pandas as pd
from scipy import ndimage

def load_and_preprocess(dicom_path):
    ds = pydicom.dcmread(dicom_path)
    image=ds.pixel_array
    
    hu_image = image * ds.RescaleSlope + ds.RescaleIntercept
    np.clip(hu_image, -1000, 400, out=hu_image)
   
   
    return hu_image
    

def segment_lungs(hu_image):
    threshold = -400
    binary_image = hu_image < threshold
    cleaned_image=morphology.erosion(binary_image)
    cleaned_image=morphology.dilation(cleaned_image)
    labeled_image=measure.label(cleaned_image)
    regions=measure.regionprops(labeled_image)
    
    sorted_regions=sorted(regions, key=lambda x: x.area, reverse=True)
    top_2=sorted_regions[:2]
   
    lung_mask=np.zeros_like(labeled_image)
    for region in top_2:
        lung_mask[labeled_image==region.label]=1
    return lung_mask
    

def find_nodule_candidates(hu_image, lung_mask):
    
    filled_lung_mask = ndimage.binary_fill_holes(lung_mask)
    candidate_mask = filled_lung_mask.astype(int) - lung_mask.astype(int)
    labeled_candidates=measure.label(candidate_mask)
    candidate_regions=measure.regionprops(labeled_candidates, intensity_image=hu_image)
    
    candidates_info=[]
    for region in candidate_regions:
       if region.area < 50 or region.area > 400:
          continue
       
       y,x = region.centroid
       candidates_info.append({'x': x, 'y': y, 'area': region.area, 'mean_intensity': region.mean_intensity})
    candidates_df = pd.DataFrame(
        candidates_info,
        columns=['x', 'y', 'area', 'mean_intensity']
)
    return candidates_df
    

def run_pipeline(dicom_path):
    hu_image=load_and_preprocess(dicom_path)
    lung_mask=segment_lungs(hu_image)
    candidates_df=find_nodule_candidates(hu_image, lung_mask)
    return hu_image, lung_mask, candidates_df

if __name__=='__main__':
    hu_image, lung_mask, candidates_df=run_pipeline('data/raw/sample.dcm')
    print(f"Found {len(candidates_df)} nodule candidates:")
    

    candidates_df.to_csv('outputs/nodule_candidates.csv', index=False)
    print(candidates_df)

    plt.figure(figsize=(15,5))
    plt.subplot(1,3,1)
    plt.imshow(hu_image, cmap='gray')
    plt.title('HU Image')
    

    plt.subplot(1,3,2)
    plt.imshow(lung_mask, cmap='gray')
    plt.title('Lung Mask')
    

    plt.subplot(1,3,3)
    plt.imshow(hu_image, cmap='gray')

    if not candidates_df.empty:
     plt.scatter(
        candidates_df['x'],
        candidates_df['y'],
        c='red',
        s=50,
        marker='x'
     )
    else:
     print("No candidates to plot.")
    plt.title(f'{len(candidates_df)} Candidates found')
    plt.savefig('outputs/day7_mini_project.png')
    plt.show()
    