# GeometricDL


Informed sampling layers for PointNet++.
This library implements various methods and processes to sample points from a point cloud according to principal curvature of the neighbourhood of points, eigen-entropy, omnivariance, planarity and shericity.
It further allows combination of different sampling methods. For example we can still use farthest-point sampling from PointNet++, but bias it towards points with high scores on some metric.

Alternative approaches include preprocessing and subsampling the point-cloud before. Then we can still benefit from the RF of the farthest-point sampling.
