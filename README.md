# Digital Image Processing: Edge Detection, Region Segmentation & Scale-Space Features

Implementation of fundamental computer vision algorithms for interactive contour extraction, graph-based region segmentation, and scale-space feature detection from scratch using Python, NumPy, and SciPy, evaluated on the [Berkeley Segmentation Dataset (BSDS300)](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300/html/dataset/images.html).

---

## Repository Structure

```text
.
├── part1/                      # Part 1: Intelligent Scissors & Marr-Hildreth
│   ├── image1/                 # Input image, feature maps, distance maps, segmentations
│   ├── part1.py                # Standalone Python implementation
│   ├── part1.ipynb             # Jupyter Notebook
│   └── report.tex              # Comprehensive LaTeX report
│
├── part2/                      # Part 2: Region Segmentation & GrabCut
│   ├── image2/                 # Input image, likelihood maps, WCSS curves, cut masks
│   ├── part2.py                # Standalone Python implementation
│   ├── part2.ipynb             # Jupyter Notebook
│   └── report.tex              # Comprehensive LaTeX report
│
├── part3/                      # Part 3: Scale-Space Keypoints & SIFT Features
│   ├── image3/                 # Scale-space pyramids, corner maps, matching results
│   ├── part3.py                # Implementation script (in progress)
│   └── report.tex              # LaTeX report
│
└── README.md
```

---

## Overview of Assignment Parts

### Part 1: Interactive Segmentation via Intelligent Scissors (Live-Wire)
Builds up to the **Intelligent Scissors** technique (Mortensen & Barrett, 1995) by formulating interactive boundary tracing as a shortest-path problem on an 8-connected pixel lattice:
- **Marr-Hildreth Edge Detector:** Implements Laplacian of Gaussian (LoG) convolution $\nabla^2(I * G_\sigma)$, 8-connected zero-crossing detection, and slope thresholding. Includes an empirical study of the Gaussian smoothing radius ($\sigma$) and slope threshold ($t$).
- **Graph Cost Formulation:** Constructs an 8-connected weighted planar graph combining three complementary local costs:
  1. *Zero-Crossing Cost ($c_z$):* Inverse LoG zero-crossing jump strength.
  2. *Gradient Magnitude Cost ($c_g$):* Inverse Sobel gradient magnitude.
  3. *Gradient Direction Cost ($c_o$):* Directional penalty enforcing orthogonality between image gradient normal and movement tangent ($\hat{\mathbf{g}} \cdot \hat{\mathbf{d}} = 0$).
- **Dijkstra's Algorithm & Distance Maps:** Computes cumulative distance maps from seed vertices via a min-heap priority queue, visualizing wavefront propagation under four cost configurations (zero-crossings only, magnitude only, orientation only, and combined).
- **Interactive Segmentation:** Traces piecewise optimal shortest paths between consecutive user-selected seeds, closing the contour and rasterizing binary segmentation masks.
- **Cost Function Modifications:** Explores two architectural variants:
  - Replacing Marr-Hildreth zero crossings with **Canny edge detection** (non-maximum suppression + hysteresis).
  - Rebalancing the cost weight vector ($w_z = 0.8, w_g = 0.1, w_o = 0.1$) to prioritize mathematical zero-crossing fidelity.

---

### Part 2: User-Guided Region Segmentation & GrabCut
Explores region-based segmentation techniques leading up to the **GrabCut** algorithm (Rother et al., 2004), evaluated on BSDS300 image #113044 (grazing horses):
- **(a) Maximum Likelihood Classification:** Fits 3D multivariate Gaussian distributions $\mathcal{N}(\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$ with ridge covariance regularization to marked foreground and background color subrectangles. Visualizes normalized likelihood maps and classifies pixels via maximum likelihood.
- **(b) Constrained K-means Clustering:** Implements 2-class K-means initialized from empirical seed means while locking user-marked pixels to prevent cluster drift. Tracks Within-Cluster Sum of Squares (WCSS) descent to convergence across iterations.
- **(c) Graph Cuts via Min-Cut / Max-Flow:** Solves binary Markov Random Field (MRF) energy minimization on a 4-connected grid using SciPy's `maximum_flow`:
  - *Regional penalties (t-links):* Negative log-likelihoods from the Gaussian appearance models with hard seed constraints.
  - *Boundary penalties (n-links):* Contrast-dependent edge weights $V_{p,q} = \lambda \exp(-\|\mathbf{x}_p - \mathbf{x}_q\|^2 / 2\sigma^2)$.
  - *Parameter study:* Explores under- and over-regularization across variations in $\lambda$ and $\sigma$.
- **(d) GrabCut-like Iterative Segmentation:** Eliminates manual seed labeling by taking only a single user-drawn bounding box. Alternates between:
  1. *E-step:* Min-cut graph cut partitioning within the box.
  2. *M-step:* Gaussian model re-estimation from newly assigned foreground/background pixels.

---

### Part 3: Scale-Space Keypoints & SIFT Feature Extraction
Implements scale-space blob detection and corner extraction in the manner of SIFT (Lowe, 2004):
- **Scale-Space LoG Extrema:** Builds a multi-scale representation of the image by convolving with scale-normalized Laplacian of Gaussian filters across continuous octaves. Detects local extrema in 3D $(x, y, \sigma)$ space with subpixel quadratic interpolation.
- **Harris-Stephens / Shi-Tomasi Corner Detector:** Computes spatial image gradients ($I_x, I_y$), builds the second-moment structure tensor $M$, and visualizes the determinant, trace, and cornerness response function $R = \det(M) - k\,\text{tr}(M)^2$ alongside detected corner overlays.
- **SIFT Keypoint Filtering:** Refines raw scale-space extrema by:
  - Rejecting low-contrast candidates below an intensity contrast threshold.
  - Suppressing unstable edge responses using the ratio of principal curvatures from the Hessian matrix (Hessian cornerness criterion).
- **Geometric & Photometric Invariance Evaluation:** Applies affine transformations (rotation, scaling), illumination shifts, and additive Gaussian noise to the test image, comparing transformed keypoints against original detections to evaluate feature repeatability and localization stability.

---

## Requirements & Setup

### Prerequisites
- Python 3.9+
- Recommended virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy scipy matplotlib opencv-python pillow
```

### Running the Code

To execute Part 1 (Intelligent Scissors):
```bash
cd part1
python3 part1.py
```

To execute Part 2 (GrabCut & Region Segmentation):
```bash
cd part2
python3 part2.py
```

All generated output images, intermediate progression plots, and segmentation masks will automatically be saved into their respective `image1/` and `image2/` directories.

