# Eigen Faces

This is a program that trains set of clusters to reconstruct faces from the LFW Face dataset. This program also demonstrates the usage of PCA to reduce the dimensionality of data, and the differences between the number of principle components used to reconstruct a face.

## Usage

`python3 faces.py`

## Dependencies

- `python 3.8+`

### Python Dependencies

- `numpy`
- `matplotlib`
- `sklearn`

## Results

### PC1

![](https://github.com/is386/eigenfaces/blob/main/images/pc1.png?raw=true)

### First face with PC1

![](https://github.com/is386/eigenfaces/blob/main/images/reconstruction1.png?raw=true)

### First face with 95% accuracy

- `Number of Components: 189`

![](https://github.com/is386/eigenfaces/blob/main/images/reconstruction2.png?raw=true)

### Cluster Centers

![](https://github.com/is386/eigenfaces/blob/main/images/centers.png?raw=true)

### Cluster Mins

![](https://github.com/is386/eigenfaces/blob/main/images/min.png?raw=true)

### Cluster Maxes

![](https://github.com/is386/eigenfaces/blob/main/images/max.png?raw=true)
