Machine Learning Operations with ConvNeXt2: A Case Study in Classification of Animal Images 
==============================
[<img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white">](https://www.youtube.com/watch?v=dQw4w9WgXcQ?autoplay=1)
[<img src="https://img.shields.io/badge/Weights_&_Biases-FFBE00?style=for-the-badge&logo=WeightsAndBiases&logoColor=white">](https://scontent-arn2-2.xx.fbcdn.net/v/t1.15752-9/324219590_702995398045880_3444596723210508741_n.jpg?_nc_cat=108&ccb=1-7&_nc_sid=ae9488&_nc_ohc=Ib-CcSC91PUAX-bZQZQ&_nc_ht=scontent-arn2-2.xx&oh=03_AdTxAZGsuouCqphrNQsysFSHP01yAha4iFyapgjLQT7_qA&oe=63E77E24)
[<img src="https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue">](https://i.redd.it/arqzi89s0q9a1.jpg)
[<img src="https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white">](http://vafler.dk/)
[<img src="https://img.shields.io/badge/GoogleCloud-%234285F4.svg?style=for-the-badge&logo=google-cloud&logoColor=white">](https://i.redd.it/hyyapbqpp3v91.jpg)
[<img src="https://img.shields.io/badge/PyTorch%20Lightning-792DE4?style=for-the-badge&logo=pytorch-lightning&logoColor=white">](https://miro.medium.com/max/500/1*qHbAsMNmdWQJkzm2SUA-8w.jpeg)

[![codecov](https://codecov.io/gh/rreezN/dtu_mlopsproject/branch/main/graph/badge.svg?token=DW6XAXYSZR)](https://codecov.io/gh/rreezN/dtu_mlopsproject)
![example workflow](https://github.com/rreezN/dtu_mlopsproject/actions/workflows/codecoverage.yml/badge.svg)
![example workflow](https://github.com/rreezN/dtu_mlopsproject/actions/workflows/tests.yml/badge.svg)
![example workflow](https://github.com/rreezN/dtu_mlopsproject/actions/workflows/flake8.yml/badge.svg)
![example workflow](https://github.com/rreezN/dtu_mlopsproject/actions/workflows/isort.yml/badge.svg)

## Overall goals of the project
<p align="center">
  <img align="right" src="pictures/wide_animals_drawing.png" alt="drawing" width="450"/>
</p>
The goal of this project is to use the image classification model [ConvNeXt V2](https://arxiv.org/abs/2301.00808) to classify a 10 class animal data set from [Animals - V2](https://www.kaggle.com/datasets/utkarshsaxenadn/animal-image-classification-dataset).

Our group (42) consists of:

David Ari Ostenfeldt, s194237\
Kristian Rhindal Møllmann, 194246\
Dennis Chenxi Zhuang, s194247\
Kristoffer Marboe, s194249\
Kasper Niklas Kjær Hansen, s194267

## What framework are you using?
For this project the [Pytorch Image Models](https://github.com/rwightman/pytorch-image-models) framework is used. This framework implements state-of-the-art image models, and is a good fit for the goals of the project. 

## How do you intend to include the framework in your project?
Pytorch Image Models contains hundreds of pretrained state of the art models, which are fit to use in our project. Initially, we will use a pre-trained model in our pipeline. Using the pre-trained model will allow us to focus on implementing the various techniques taught in the course. If possible and time permits the model can be further improved for our data set.

## What data are you going to run on (initially)?
For the project we are going to be working with the [Animals V2 Image Classification Data Set](https://www.kaggle.com/datasets/utkarshsaxenadn/animal-image-classification-dataset). The data set consists of 10 classes, with each class containing 2000 training images, 100 validation images, a varying amount of testing images and 6 interesting images. This means there are a total of 20000 training images and 1000 validation images. 

| Class       |  Cat |  Cow |  Dog | Elephant | Gorilla | Hippo | Monkey | Panda | Tiger | Zebra |
|-------------|-----:|-----:|-----:|---------:|--------:|------:|-------:|------:|------:|------:|
| Training    | 2000 | 2000 | 2000 |     2000 |    2000 |  2000 |   2000 |  2000 |  2000 |  2000 |
| Validation  |  100 |  100 |  100 |      100 |     100 |   100 |    100 |   100 |   100 |   100 |
| Testing     |  394 |  177 |   88 |      306 |      30 |    57 |    184 |   237 |   164 |   270 |
| Interesting |    6 |    6 |    6 |        6 |       6 |     6 |      6 |     6 |     6 |     6 |

The images vary in size and thus need to be transformed to a standard size.

## What deep learning models do you expect to use?
We intend to use the model [ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders](https://arxiv.org/abs/2301.00808), which is a recent update to the original ConvNeXt. The model contains pre-trained models of different sizes, of which we will be using the pretrained 3.7 million parameter model, Atto.

--------
See [checklist](checklist.md).


<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. </small></p>
# cookiecutterdatascience </p>
# dtu_mlopsproject
