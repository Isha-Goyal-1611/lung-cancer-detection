import matplotlib.pyplot as plt
import pydicom
ds=pydicom.dcmread('data/raw/sample.dcm', force=True)

print(f"patient id: {ds.PatientID}")
print(f"Patient Name:{ds.PatientName}")
print(f"Patient Age:{ds.PatientAge}")
print(f"Patient Sex:{ds.PatientSex}")

pixel_array=ds.pixel_array
print(f"Pixel Array Shape: {pixel_array.shape}")
print(f"Pixel Array Data Type: {pixel_array.dtype}")

plt.imshow(pixel_array,cmap='gray')
plt.title('Raw CT Scan - Before Normalization')
plt.colorbar()
plt.savefig('outputs/day2_raw_ct_scan.png')
plt.show()