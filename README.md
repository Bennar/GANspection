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

The structure allows us to do reconstructions of real samples, ** G(E(x)) **,

![Recon](https://user-images.githubusercontent.com/35339379/129281354-29dd6b47-4c44-4490-8796-b0f7a81d364f.png)

## Anomaly Score
\alpha

## Conclusion
