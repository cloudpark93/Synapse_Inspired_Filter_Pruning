# Synapse_Inspired_Filter_Pruning
In the human synaptic system, there are two important channels known as excitatory and inhibitory neurotransmitters that transmit a signal from a neuron to a cell. Adopting the neuroscientific perspective, we propose a synapse-inspired filter pruning method.

* This is much advanced than my other repository, [easy-filter-pruning](https://github.com/cloudpark93/easy-filter-pruning).

# co-researcher
Please visit my [**co-researcher's**](https://github.com/jinsoo9595) github as well!  
https://github.com/jinsoo9595

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

The below attached image is the sensitivity analysis of ResNet18 with three different methods.  
It is noticeable that different methods have yielded differernt sensitivity patterns.  
![sensitivity method별 다른거](https://user-images.githubusercontent.com/78515689/161476387-0a3ddae7-ecba-4a9b-a962-4266ebd0e09c.PNG)

Sensitiity analysis
# Pruning Results

# Results Analysis
