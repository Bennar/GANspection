# GANspection
#### *A semi-supervised one-class anomaly detection method based on a generative adversariel network structure*
##### (Work-in-progress)

Anomaly detection on images is a fundamental computer vision task and are of great interest in an industrial production setting, where products have to be quality assured and the magnitude of the production does not allow for human supervision. In recent years deep learning has improved performance in this field and research have been done on a variety of network structures and methods to determine an anomaly. Especially using generative models, such as autoencoders or GANs, have shown promising results. In this setting, the idea is, at training time, to have the generative models learn the representation of the normal-data and then at test time extract some embedding, reconstruction or latent representation of a new sample, and based on this calculate an anomaly score.

## The Data: MVTec AD
[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) is a dataset for benchmarking anomaly detection methods with a focus on industrial inspection. It contains over 5000 high-resolution images divided into fifteen different object and texture categories. Each category comprises a set of defect-free training images and a test set of images with various kinds of defects as well as images without defects.

To scope my project, i will focus on the *hazelnut* images.

![Hazelnut_data](https://user-images.githubusercontent.com/35339379/129034371-315f038a-b1fe-4543-8753-e1fa59b0e1a0.png)


## The Model: BiGAN/ALI

![BiGAN](https://user-images.githubusercontent.com/35339379/128866480-17861056-13e5-4e81-9909-50f13f6f6649.png)
## Anomaly Score

## Conclusion
