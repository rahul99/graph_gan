# Project Architecture and Process flow

## Entry point:
* driver.py is the entry point to the project. All components are integrated and called from this script

## Graph extraction
* We use the idea as proposed by "Alejandro Newell and Jia Deng - [Pixels to graphs by associative embedding](https://arxiv.org/abs/1706.07365)
      
## Identify Target Image(and target graph):
   * Generate target image by altering the adjacency matrix in the artoon image generation utility for example
   * Cartoon images released by Devi Parikh's group in VT can be augmented to tailor as per our need. They have rendered 10 target images per condition image. They also maintain subject-predicate-object relationship that we can augment as graph adjacency matrix

## Graph Embedding
* For graph representation, run the method get_graph_as_images() defined in Graph2ImgEmbedding.py within embedding module
* This script does the following:
    * Get a graph adjacency matrix
    * Perform LLE embedding as implemented by "Palash Goyal and Emilio Ferrara. Graph embedding techniques, applica-tions, and       performance: A survey"
    * Store the top-k eigen vectors of this matrix
    * Form k/2 2D planes using a pair of eigen vectors in the successive order
    * Project the adjacency matrix onto each of these hyper planes
    * Descritize the distribution of points in the plane in a 2D grid of the same size as the input image
    * Each entry of the gird corresponds to the number of points falling on that grid
    * The resultant stack of 2D frames in the final representation of the target graph on which our Generator model will be           conditioned
   
## Model training
U-Net_like architecture for Generator and CNN for discriminator

## Reference code

* [Collection of generative models, e.g. GAN, VAE in Pytorch and Tensorflow](https://github.com/wiseodd/generative-models)

* [adversarial example library for constructing attacks, building defenses, and benchmarking both](https://github.com/tensorflow/cleverhans)
