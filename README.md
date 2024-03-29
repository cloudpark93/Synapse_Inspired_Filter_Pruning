# Synapse_Inspired_Filter_Pruning
In the human synaptic system, there are two important channels known as excitatory and inhibitory neurotransmitters that transmit a signal from a neuron to a cell. Adopting the neuroscientific perspective, we propose a synapse-inspired filter pruning method.

* This is much advanced than my other repository, [easy-filter-pruning](https://github.com/cloudpark93/easy-filter-pruning).

# co-researcher
Please visit my [**co-researcher's**](https://github.com/jinsoo9595) github as well!  
https://github.com/jinsoo9595

# Development time
Dec 2020 ~ May 2021

# Preprint in arXiv
The preprint of this work, [D-Score: A Synapse-Inspired Approach for Filter Pruning](https://arxiv.org/abs/2308.04470), has been uploaded to arXiv on 8th Aug 2023.  



# Some words for the visitors
For our case, we initially began this project with different development environments for VGGnets(CIFAR-10) and ResNets(ImageNet).  
I can proudly say my [co-researcher](https://github.com/jinsoo9595) and I developed from scratch for VGGnets with CIFAR-10 dataset.  
For ResNets (with ImageNet dataset), we used [**jkjung-avt's github**](https://github.com/jkjung-avt/keras_imagenet) as the reference, and tried to adapt his **keras_imagenet** code into ours.  

So please do visit [jkjung-avt's github](https://github.com/jkjung-avt/keras_imagenet) too!

I tried my best to combine the VGGnets(CIFAR-10) pruning codes and ResNets(ImageNet) pruning codes as much as possible.

For the main pruning mechanisms of various ***Synapse inspired filter pruning***, please refer to the three scripts in /utils/

```
Synapse_Inspired_Filter_Pruning/
│   ...
└───utils/
    │   pruning_method_conv.py
    │   pruning_method_fc.py
    │   geometric method.py
    │   ...
```


# Main Concept of Synapse Inspired Filter Pruning
<img src = "https://user-images.githubusercontent.com/78515689/161383050-4ff6ab9b-88c7-4c37-99fc-ea0fcd245f21.png" width="800px" height="350px">

In the synaptic transmission system of a biological neural network, the basis of an artificial neural network, there coexists excitatory and inhibitory neurotransmitters. The excitatory neurotransmitter enhances or increases the activation in the postsynaptic membrane, while the inhibitory neurotransmitter decreases or prevents the activation. Similarly, filters of CNN models are composed of positive weights and negative weights. Considering the neuroscientific perspective, we propose a new filter pruning approach that separately analyzes the positive and negative weights in the filters.

# Requirements
* Python 3.8
* TensorFlow 2.2.0 (GPU, too)
* Keras 2.4.3
* [**Keras-surgeon 0.2.0**](https://github.com/BenWhetton/keras-surgeon)
* Keras_flops 0.1.2
* pandas 1.2.3
* numpy 1.19.4
* matlplotlib 3.4.1

Thanks to [**BenWhetton**]((https://github.com/BenWhetton/keras-surgeon)) for developing such a useful and convenient pruning tool in Keras.  
Please do visit his website for the details, too!  
https://github.com/BenWhetton/keras-surgeon

# Filter Ranking Methods
*For the details, please refer to the attached manuscript, **Section 3.1**!*  

**1. Dynamic Score (Major concept)**
* Assigns scores to positive and negative weights in the filters according to their values, and ranks the importance of the filters by their overall scores.  

**2. Dynamic Step (Applied concept 1)**
* Assigns scores to positive and negative weights in the filters according to their values, and ranks the importance of the filters by their simultaneous importance.

**3. Dynamic Step with Geometric Median (Applied concept 2)**
* Adopted the idea suggested by [**YangHe**](https://github.com/he-y/filter-pruning-geometric-median)
* Ranks the importance of the filters by the Euclidean distances of positive and negative filters, from the shortest to the longest.

# Pruning Process  
*For the details, please refer to the attached manuscript, **Section 3.2 ~ 3.3**!*  
1. Filter Importance  
    * Determine the importance of filters by any of the three suggested methods.  
2. Sensitivity Analysis  
    * Determine the sensitivity of individual layers to pruning.  
    * Iteratively prune each layer and evaluate the accuracy of the pruned network in every step..  
3. Parallel Pruning 
    * Set a pruning threshold accuracy (aka target accuracy) based on the pruning sensitivity analysis.  
    * Calculate the number of filters to be eliminated in all applicable layers.  
        * Differernt number of filters are eliminated in each layer (refer to the image below or the attached manuscript for better understanding)  
        * <img src = "https://user-images.githubusercontent.com/78515689/161475388-d8ecc0e1-be6b-4178-9b78-ae8134b6329a.PNG" width="500px" height="320px">  
4. Retraining  
    * Retrain the pruned network for fewer epochs than the origninal model.

\* **The below attached image is the sensitivity analysis of ResNet18 with three different pruning methods.**  
\* **It is noticeable that different methods have yielded different sensitivity patterns.**  
![sensitivity method별 다른거](https://user-images.githubusercontent.com/78515689/161476387-0a3ddae7-ecba-4a9b-a962-4266ebd0e09c.PNG)  

# Pruning Results
The table below is the comparison of our pruning methods and other methods for VGG-16 trained on CIFAR-10 dataset.  
Other more interesting results on ResNet18, 34, 50 trained on ImageNet dataset are in the manuscript!  

Model|Approach|Acc Drop(%)|Params Reduction(%)|FLOPs Reduction(%)
:----------:|:--------:|:---:|:---:|:---:
VGG-16|PFEC|**-0.15**|64.0|34.2
VGG-16|[FPGM](https://github.com/he-y/filter-pruning-geometric-median)|0.04|-|34.2
VGG-16|NSP|-0.04|-|54.0
VGG-16|[HRank](https://github.com/lmbxmu/HRank)|0.53|82.9|53.5
VGG-16|NS|-0.14|**88.52**|51.0
VGG-16|Ours (D-Score)|0.16|87.03|64.81
VGG-16|Ours (D-Step)|0.12|86.70|**65.40**
VGG-16|Ours (D-Step GM)|-0.10|97.17|64.37

# Feature Map Visualization
Feature maps are visualized and compared with other method in order to validate the functionality of our proposed methods.
* Model: VGG-16  
* Dataset: CIFAR-10  
* Visualization layer: 2nd Conv layer  
* Pruning amount: 20% of filters to be pruned
* Input image: Frog (below)
* <img src = "https://user-images.githubusercontent.com/78515689/161480185-3f71d978-71be-41fa-a141-278c8247aeb2.JPEG" width="200px" height="151px">  

The image below is an additional explanation for how our figure is composed/interpreted.  
![visualize 설명](https://user-images.githubusercontent.com/78515689/161481705-ec36f9ed-0e42-4751-9287-062cf9bbdaf3.PNG)  

The image below shows that when the filter importance is ranked by our method (D-Score), it does not prune the unique feature map producing filters (eg, filter 4, and filter 17).
![featuremap visualzie](https://user-images.githubusercontent.com/78515689/161479986-ce24e31e-d7a0-47f6-bb76-14bdfe20d9d3.PNG)  

