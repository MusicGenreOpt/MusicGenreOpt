# Optimization Based Ensemble Methods for Music Genre Classification

This repository contains an implementation of mathematical models and algorithms for the music genre classification problem, as proposed in the paper ["Optimization Based Ensemble Methods for Music Genre Classification"]().

## Abstract

With the increasing popularity of music streaming services and the abundance of song variety, the need for automated tools in the music industry is more than ever. This paper proposes an optimization-based ensemble approach to the automatic music genre classification problem. We particularly design solutions that leverage heterogeneity across multiple segments of a song. To that effect, we split each song into disjoint segments and use segment-specific genre classification probability for each segment of each song. Then, the proposed optimization algorithms aim to find the best set of weights in combining the genre classification probabilities of segments into an aggregated class probability for the entire song. We use a publicly available dataset and compare the performances of the proposed algorithms to those of simple and complex ensemble methods. The results reveal that our solutions increase (decrease) the classification accuracy by approximately 3-6% (2.33%) when compared to simple (complex ensemble) benchmarks. Our proposed methods provide a good compromise between model accuracy and model interpretability. Finally, they also perform well when used for multi-genre classification tasks. 

