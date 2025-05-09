# Pointnet3

## Overview
Pointnet3 is a Python library that enhances point cloud sampling for PointNet++ by incorporating informed sampling layers. It provides methods to sample points based on geometric properties such as principal curvature, eigen-entropy, omnivariance, planarity, and sphericity. The library supports flexible combinations of sampling techniques, allowing users to bias traditional methods like farthest-point sampling towards points with high geometric significance. Additionally, it offers preprocessing and subsampling capabilities to optimize point cloud data while preserving the benefits of robust sampling strategies.

## Features
- **Informed Sampling**: Sample points based on geometric metrics including principal curvature, eigen-entropy, omnivariance, planarity, and sphericity.
- **Hybrid Sampling**: Combine multiple sampling methods, such as biasing farthest-point sampling with geometric scores.
- **Preprocessing Support**: Subsample point clouds prior to processing to improve efficiency while maintaining geometric integrity.
- **Extensible Framework**: Easily integrate new sampling metrics or methods into the existing pipeline.

## Installation
To install Pointnet3, clone the repository and install the required dependencies:

```bash
git clone https://github.com/PaulHosek/Pointnet3.git
cd Pointnet3
pip install -r requirements.txt
```

Ensure you have Python 3.8+ and the necessary dependencies listed in `requirements.txt`.

## Usage
Below is a basic example of using Pointnet3 to sample points from a point cloud with curvature-biased farthest-point sampling:

```python
from pointnet3.sampling import CurvatureBiasedFPS

# Load your point cloud data
point_cloud = load_point_cloud("path/to/point_cloud.ply")

# Initialize the sampler
sampler = CurvatureBiasedFPS(num_samples=1024)

# Sample points
sampled_points = sampler.sample(point_cloud)

# Use sampled points with PointNet++ or other downstream tasks
```

