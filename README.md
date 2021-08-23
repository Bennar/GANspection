# GANspection
#### *A semi-supervised one-class anomaly detection method based on a generative adversariel network structure*
##### (Work-in-progress)

Anomaly detection on images is a fundamental computer vision task and are of great interest in an industrial production setting, where products have to be quality assured and the magnitude of the production does not allow for human supervision. In recent years deep learning has improved performance in this field and research have been done on a variety of network structures and methods to determine an anomaly.  Utilizing generative models, such as autoencoders or GANs, have shown promising results. In this setting, the idea is, at training time, to have the generative models learn the representation of the normal-data and then at test time extract some embedding, reconstruction or latent representation of a new sample, and based on this calculate an anomaly score.

## The Data: MVTec AD
[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) is a dataset for benchmarking anomaly detection methods with a focus on industrial inspection. It contains over 5000 high-resolution images divided into fifteen different object and texture categories. Each category comprises a set of defect-free training images and a test set of images with various kinds of defects as well as images without defects.

To exemplify, i will focus on the *hazelnut* images.

![Hazelnut_data](https://user-images.githubusercontent.com/35339379/129034371-315f038a-b1fe-4543-8753-e1fa59b0e1a0.png)


## The Model: BiGAN/ALI
The model i use is a GAN-like structure named BiGAN or ALI. It was first suggested in in two seperate papers in 2016 [ [1](https://arxiv.org/abs/1605.09782v7) [2](https://arxiv.org/abs/1606.00704) ]. The structure resembles a typical GAN, but in addition to the generator and discriminator networks, it features an encoder network, that takes a real sample and encode it to a vector of the same size as the generator input. The discriminator is fed pairs of image and feature representation. All networks are CNNs. The BiGAN is trained with wasserstein-distance as it is widely shown to stabalize GAN training, along with a series of GAN stabalizing tricks from [Salimans et. al. 2016](https://arxiv.org/abs/1606.03498). As a novelty I add *l2 loss* between the feature representation of real and generated samples whrn updating the generator and encoder network. This has shown to improve performance when doing anomaly scoring. The feature representations for the generated examples, **z**, are drawn from a *standar normal distribution*.

![BiGAN](https://user-images.githubusercontent.com/35339379/128866480-17861056-13e5-4e81-9909-50f13f6f6649.png)

As with a normal GAN, we can generate fake samples. The generator was trained with feature representations sampled from a standard normal distribution, so we can see what happens when you draw from outside this distribution.

![Gen](https://user-images.githubusercontent.com/35339379/129357397-a80f8c90-3667-4f29-a9e6-8105fc991af9.png)

The structure allows us to do reconstructions of real samples, **G(E(x))**,

![Recon](https://user-images.githubusercontent.com/35339379/129281354-29dd6b47-4c44-4490-8796-b0f7a81d364f.png)

## Anomaly Score
With the task of determining whether a sample is an anomaly or not, I propose an Anomaly Score consisting of 3 terms. First, the l2 norm of the feature reprsentation. Since, at training, the generator is sampling from a standard normal distribution and the encoder is trained to transform a normal sample to the feature representation, we would expect the l2 norm to be equal to zero for normal samples, where as an anomaly would skew the distribution and therefore get something different from zero.
 
 Second we use the l1 norm of the reconstruction error. The idea is that the networks have learned only transformation of normal features, thus when reconstructing an abnormal sample, it will not be able to reconstruct the abnormality. The difference between the sample and the abnormality-free-reconstruction is then non-zero around a abnormality. This method is often used in other similar setups, and especially when segmentation is also needed.
 
 third we use the l2 distance of the second-to-last layer in the discriminator, of inputs of sample and feature representation and reconstruction and feature reprsentation. This is motivated by results from other research articles and the thought that this embedding will show divergence beacause of the failed encoding and generation of the model.

![hist](https://user-images.githubusercontent.com/35339379/129928744-f653d25c-2781-416d-9ecb-47f76ef1f3d7.png)

The terms are weighted, and the collected anomaly score is then threshold to determine whether a sample is abnormal or not. These are hyperparameters to be determined.

**A = α ||E(x)||<sub>2</sub> + β ||x - G(E(x))||<sub>1</sub> + γ ||f(x,E(x)) - f(G(E(x)),E(x))||<sub>2</sub>**

![ROC](https://user-images.githubusercontent.com/35339379/129928846-785f0045-1101-4f3a-89a1-e4ad8bd95a83.png)

| Measure        | auROC           |
| ------------- |:-------------:|
| Feature L2     | 0.89 |
| Reconstruction L1      | 0.71     |
| Discriminator feature L2 | 0.60      |
| Anomaly Score | 0.83      |

## Conclusion
With the objective of doing semi-supervised anomaly detection the GANspection method succeds. Yet, with an average auROC over the MVTec dataset, GANspection does not perform on the same level as other top methods [(see "detection auroc")](https://paperswithcode.com/sota/anomaly-detection-on-mvtec-ad). However, compared to other top-performing-models GANspection does not require pretrained networks or human-selected-features as most of them do. In addition, GANspection is also a generative model, allowing for synthesization of data, and are as far as can be found, the best performing generative model.

Futher work with the GANspection method would include a larger hyperparameter optimization, since not much have been done here.


## Install and run on your setup
(Work in progress)
