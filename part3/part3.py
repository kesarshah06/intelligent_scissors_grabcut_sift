import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.ndimage as ndimage
from PIL import Image
from pathlib import Path

filename = Path(__file__).with_name("input.jpg")

#.....................................................processing the input image..............................................
img = Image.open(filename).convert('L')                                            

max_size = 1024                                                                    # making sure the image is not too large for processing

scale = min(1.0, max_size / max(img.size))
new_size = (                                                                       # making the image 1024*1024 or smaller if it is already smaller than that
    int(img.size[0] * scale),
    int(img.size[1] * scale)
)

img = img.resize(new_size)                                                         # resizing the image itself

image = np.asarray(img, dtype=np.float64)                                          # converting the image to a floating point array

#.....................................................Generating the LoG (Laplacian of Gaussian) kernel..............................................
def log_kernel(sigma):
    
    size = int(6 * sigma)                                                          # Kernel size is typically 6 times the standard deviation
    if size % 2 == 0:
        size += 1                                                                  # Ensure the kernel size is an odd number
        
    ax = np.arange(-(size // 2), size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    
    kernel = (xx**2 + yy**2 - 2 * sigma**2) / (sigma**4)                           # Compute the Laplacian portion
    kernel *= np.exp(-(xx**2 + yy**2) / (2 * sigma**2))                            # Multiply by the Gaussian portion
    
    kernel = kernel - np.mean(kernel)                                              # Ensure kernel sums to zero to avoid response in flat regions
    return kernel

#.....................................................3.a Scale-Space Representation and LoG Extrema..............................................
sigmas = [1.6 * (1.2 ** i) for i in range(10)]                                     # Define scales: initial sigma=1.6, k=1.2, 10 scales
scale_space = np.zeros((image.shape[0], image.shape[1], len(sigmas)))              # Initialize 3D scale-space volume

for i, sigma in enumerate(sigmas):
    kernel = log_kernel(sigma)
    
    response = ndimage.convolve(image, kernel, mode='reflect')                     # Apply LoG filter at current scale
    scale_space[:, :, i] = np.abs(response * (sigma ** 2))                         # Scale normalization: multiply by sigma^2 and take absolute response

#.....................................................Finding 3D Local Maxima in Scale-Space..............................................
def find_local_maxima_3d(scale_space, threshold):
    
    local_max = ndimage.maximum_filter(scale_space, size=3)                        # 3x3x3 maximum filter for 26-neighbor comparison
    
    maxima_mask = (scale_space == local_max) & (scale_space > threshold)           # Check if pixel is equal to local max and above contrast threshold
    
    y, x, s = np.where(maxima_mask)                                                # Get coordinates of local maxima
    return y, x, s

threshold_a = 15.0                                                                 # Arbitrary threshold for initial maxima extraction
y_extrema, x_extrema, s_extrema = find_local_maxima_3d(scale_space, threshold_a)

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(image, cmap='gray')
for i in range(len(y_extrema)):
    sigma = sigmas[s_extrema[i]]
    circle = patches.Circle((x_extrema[i], y_extrema[i]), sigma * 1.414,           # Draw circles proportional to the scale sigma
                            edgecolor='red', facecolor='none', lw=1.5)
    ax.add_patch(circle)
ax.set_title("3(a) Absolute LoG Scale-Space Extrema")
ax.axis("off")
plt.savefig("3a_log_extrema.png")
plt.close()

#.....................................................3.b Harris-Stephens / Shi-Tomasi Corner Detector..............................................
def harris_corner_detector(img, k=0.04, window_size=3):
    
    dy, dx = np.gradient(img)                                                      # Compute image gradients Ix and Iy using simple difference
    
    Ixx = dx ** 2
    Iyy = dy ** 2
    Ixy = dx * dy
    
    offset = window_size // 2
    
    det_img = np.zeros_like(img)
    trace_img = np.zeros_like(img)
    cornerness = np.zeros_like(img)
    
    Sxx = ndimage.gaussian_filter(Ixx, sigma=1.0)                                  # Apply Gaussian window function for sum of squared differences
    Syy = ndimage.gaussian_filter(Iyy, sigma=1.0)
    Sxy = ndimage.gaussian_filter(Ixy, sigma=1.0)
    
    det_img = (Sxx * Syy) - (Sxy ** 2)                                             # Determinant of structure tensor
    trace_img = Sxx + Syy                                                          # Trace of structure tensor
    
    cornerness = det_img - k * (trace_img ** 2)                                    # Harris corner response function
    
    return det_img, trace_img, cornerness

det, trace, cornerness = harris_corner_detector(image)

local_max_corners = ndimage.maximum_filter(cornerness, size=5)
corner_threshold = 0.01 * cornerness.max()                                         # Threshold at 1% of the global maximum cornerness
corner_mask = (cornerness == local_max_corners) & (cornerness > corner_threshold)
cy, cx = np.where(corner_mask)

plt.figure(figsize=(16, 10))
plt.subplot(2, 2, 1); plt.imshow(det, cmap='jet'); plt.title("Determinant"); plt.axis("off")
plt.subplot(2, 2, 2); plt.imshow(trace, cmap='jet'); plt.title("Trace"); plt.axis("off")
plt.subplot(2, 2, 3); plt.imshow(cornerness, cmap='jet'); plt.title("Cornerness (R)"); plt.axis("off")
ax4 = plt.subplot(2, 2, 4)
ax4.imshow(image, cmap='gray')
ax4.scatter(cx, cy, c='r', s=10, marker='x')                                       # Plot detected corners on original image
ax4.set_title("Detected Harris Corners")
ax4.axis("off")
plt.savefig("3b_harris_corners.png")
plt.close()

#.....................................................3.c SIFT Keypoint Filtering (Contrast and Edge criteria)..............................................
def is_edge_response(img, x, y, r_thresh=10):
    
    dy, dx = np.gradient(img)                                                      # Compute gradients for Hessian matrix
    dyy, dyx = np.gradient(dy)
    dxy, dxx = np.gradient(dx)
    
    Hxx = dxx[y, x]
    Hyy = dyy[y, x]
    Hxy = dxy[y, x]
    
    tr = Hxx + Hyy                                                                 # Trace of Hessian
    det = (Hxx * Hyy) - (Hxy ** 2)                                                 # Determinant of Hessian
    
    if det <= 0:                                                                   # Principal curvatures have different signs, invalid
        return True
        
    curvature_ratio = (tr ** 2) / det
    threshold_ratio = ((r_thresh + 1) ** 2) / r_thresh                             # SIFT edge criteria formula
    
    return curvature_ratio > threshold_ratio                                       # Returns True if it's an edge (should be rejected)

filtered_x, filtered_y, filtered_s = [], [], []

for i in range(len(y_extrema)):
    y_pt, x_pt = y_extrema[i], x_extrema[i]
    
    is_edge = is_edge_response(image, x_pt, y_pt, r_thresh=10)                     # Filter 1: Reject edges using Hessian curvature ratio (Cornerness)
    
    contrast_val = scale_space[y_pt, x_pt, s_extrema[i]]
    is_low_contrast = contrast_val < 30.0                                          # Filter 2: Reject low contrast points
    
    if not is_edge and not is_low_contrast:
        filtered_x.append(x_pt)
        filtered_y.append(y_pt)
        filtered_s.append(s_extrema[i])

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(image, cmap='gray')
for i in range(len(filtered_x)):
    sigma = sigmas[filtered_s[i]]
    circle = patches.Circle((filtered_x[i], filtered_y[i]), sigma * 1.414, 
                            edgecolor='lime', facecolor='none', lw=2.0)
    ax.add_patch(circle)
ax.set_title("3(c) Final SIFT Keypoints (Filtered for Contrast & Edges)")
ax.axis("off")
plt.savefig("3c_sift_keypoints.png")
plt.close()

#.....................................................3.d Transformations and Matching..............................................
angle = 15                                                                         # Rotation angle in degrees
noise = np.random.normal(0, 10, image.shape)                                       # Additive Gaussian noise

transformed_image = ndimage.rotate(image, angle, reshape=False)                    # Apply rotation
transformed_image = transformed_image + noise                                      # Apply noise
transformed_image = np.clip(transformed_image, 0, 255)                             # Clip intensities to valid bounds

# Run full pipeline on transformed image (Simplified for visualization)
ts_space = np.zeros((transformed_image.shape[0], transformed_image.shape[1], len(sigmas)))
for i, sigma in enumerate(sigmas):
    kernel = log_kernel(sigma)
    response = ndimage.convolve(transformed_image, kernel, mode='reflect')
    ts_space[:, :, i] = np.abs(response * (sigma ** 2))

ty, tx, ts = find_local_maxima_3d(ts_space, threshold_a)                           # Extract transformed extrema

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(transformed_image, cmap='gray')
for i in range(len(tx)):
    sigma = sigmas[ts[i]]
    circle = patches.Circle((tx[i], ty[i]), sigma * 1.414, 
                            edgecolor='cyan', facecolor='none', lw=1.0)
    ax.add_patch(circle)
ax.set_title("3(d) Keypoints on Transformed Image")
ax.axis("off")
plt.savefig("3d_transformed_keypoints.png")
plt.close()