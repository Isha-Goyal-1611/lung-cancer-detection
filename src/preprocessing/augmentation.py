import numpy as np
import matplotlib.pyplot as plt

def random_flip(patch):
    if np.random.random()>0.5:
        patch=np.flip(patch,axis=0)
    if np.random.random()>0.5:
        patch=np.flip(patch,axis=1)
    if np.random.random()>0.5:
        patch=np.flip(patch,axis=2)
       
    return patch

def random_rotation_90(patch):
    k=np.random.randint(0,4)
    
    patch=np.rot90(patch,k,(1,2))
    return patch

def add_small_noise(patch, std=15):
    noise=np.random.normal(0,std,patch.shape)
    patch=patch+noise
    return patch

def augment_patch(patch):
    patch=random_flip(patch)
    patch=random_rotation_90(patch)
    patch=add_small_noise(patch)
    return patch

if __name__ == "__main__":
    fake_patch=np.random.randint(-1000,400,size=(32,32,32)).astype(float)
    augmented=augment_patch(fake_patch.copy())
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.imshow(fake_patch[16], cmap='gray')
    plt.title('Original patch(middle slice)')
    plt.colorbar()

    plt.subplot(1,2,2)
    plt.imshow(augmented[16], cmap='gray')
    plt.title('Augmented patch(middle slice)')
    plt.colorbar()

    plt.savefig('outputs/day_15_augmentation.png')
    plt.show()

    print("Original mean HU :" , fake_patch.mean().round(2))
    print("Augmentation mean HU:", augmented.mean().round(2))
    print("HU difference(should be small):",abs(fake_patch.mean()-augmented.mean()).round(2))
    

    