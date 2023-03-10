---
layout: default
nav_exclude: true
---

# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

where you instead should add your answers. Any other changes may have unwanted consequences when your report is auto
generated in the end of the course. For questions where you are asked to include images, start by adding the image to
the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

```markdown
![my_image](figures/<image>.<extension>)
```

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

will generate an `.html` page of your report. After deadline for answering this template, we will autoscrape
everything in this `reports` folder and then use this utility to generate an `.html` page that will be your serve
as your final handin.

Running

```bash
python report.py check
```

will check your answers in this template against the constrains listed for each question e.g. is your answer too
short, too long, have you included an image when asked to.

For both functions to work it is important that you do not rename anything. The script have two dependencies that can
be installed with `pip install click markdown`.

## Overall project checklist

The checklist is *exhaustic* which means that it includes everything that you could possible do on the project in
relation the curricilum in this course. Therefore, we do not expect at all that you have checked of all boxes at the
end of the project.

### Week 1

* [X] Create a git repository
* [X] Make sure that all team members have write access to the github repository
* [X] Create a dedicated environment for you project to keep track of your packages
* [X] Create the initial file structure using cookiecutter
* [X] Fill out the `make_dataset.py` file such that it downloads whatever data you need and
* [X] Add a model file and a training script and get that running
* [X] Remember to fill out the `requirements.txt` file with whatever dependencies that you are using
* [X] Remember to comply with good coding practices (`pep8`) while doing the project
* [X] Do a bit of code typing and remember to document essential parts of your code
* [X] Setup version control for your data or part of your data
* [X] Construct one or multiple docker files for your code
* [X] Build the docker files locally and make sure they work as intended
* [X] Write one or multiple configurations files for your experiments
* [X] Used Hydra to load the configurations and manage your hyperparameters
* [X] When you have something that works somewhat, remember at some point to to some profiling and see if
      you can optimize your code
* [X] Use Weights & Biases to log training progress and other important metrics/artifacts in your code. Additionally,
      consider running a hyperparameter optimization sweep.
* [X] Use Pytorch-lightning (if applicable) to reduce the amount of boilerplate in your code

### Week 2

* [X] Write unit tests related to the data part of your code
* [X] Write unit tests related to model construction and or model training
* [X] Calculate the coverage.
* [X] Get some continuous integration running on the github repository
* [X] Create a data storage in GCP Bucket for you data and preferable link this with your data version control setup
* [X] Create a trigger workflow for automatically building your docker images
* [X] Get your model training in GCP using either the Engine or Vertex AI
* [X] Create a FastAPI application that can do inference using your model
* [X] If applicable, consider deploying the model locally using torchserve
* [X] Deploy your model in GCP using either Functions or Run as the backend

### Week 3

* [ ] Check how robust your model is towards data drifting
* [ ] Setup monitoring for the system telemetry of your deployed model
* [ ] Setup monitoring for the performance of your deployed model
* [ ] If applicable, play around with distributed data loading
* [ ] If applicable, play around with distributed model training
* [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed

### Additional

* [X] Revisit your initial project description. Did the project turn out as you wanted?
* [X] Make sure all group members have a understanding about all parts of the project
* [X] Uploaded all your code to github

## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer:

42

### Question 2
> **Enter the study number for each member in the group**
>
> Example:
>
> *sXXXXXX, sXXXXXX, sXXXXXX*
>
> Answer:

s194237, 194246, s194247, s194249, s194267

### Question 3
> **What framework did you choose to work with and did it help you complete the project?**
>
> Answer length: 100-200 words.
>
> Example:
> *We used the third-party framework ... in our project. We used functionality ... and functionality ... from the*
> *package to do ... and ... in our project*.
>
> Answer:

We used the framework [Pytorch Image Models](https://github.com/rwightman/pytorch-image-models). This framework implements state-of-the-art image models, and is a good fit for the goals of the project. In the framework there hundreds of pretrained state of the art models, which are fit to use in our project. Initially, we will use a pre-trained model in our pipeline. Using the pre-trained model will allow us to focus on implementing the various techniques taught in the course. The models are implemented in a way that allows to specify various parameters, including but not limited to changing the number of in and out channels, regularisation parameters such as dropout and more.

## Coding environment

> In the following section we are interested in learning more about you local development environment.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Answer length: 100-200 words
>
> Example:
> *We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a*
> *complete copy of our development enviroment, one would have to run the following commands*
>
> Answer:

To manage the dependencies in our project, we used the pipreqs package. This ensured that all dependencies were accounted for while reducing the length of the requirements.txt file as much as possible. To setup working on the project, it is required to build a virtual environment, run `pip install --upgrade pip and then run python -m pip install -r requirements.txt`. Next the new user should use the installed `dvc` package to pull data from Google Storage, by running `git pull` followed by `dvc pull`. Now everything should be setup correctly and be ready to train models etc.

### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. Did you fill out every folder or only a subset?**
>
> Answer length: 100-200 words
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*
> Answer:

To adhere with the cookiecutter template, all code in the project is contained within the `src/` folder. All files which are initially present within the template have been altered, namely: `make_data.py`, `model.py`, `train_model.py` and `predict_model.py`. We also added a new file, `rename_files.py`, to `src/data/` since the raw-data from kaggle included files with spaces in the names which caused problems when locating all image files in subfolders when running `make_dataset.py` to create processed data. We also added a config-files for hydra located in a config/ folder within `src/models/`. While implementing unit-tests of the code, a `tests/` folder was created where test files for the data and model could be stored. To deploy the model using images, a `deployment/` folder was created, containing the `FastAPI` app files.

### Question 6

> **Did you implement any rules for code quality and format? Additionally, explain with your own words why these**
> **concepts matters in larger projects.**
>
> Answer length: 50-100 words.
>
> Answer:

We adhered to the pep8 standard format for python code, with the change that the maximum line length is 100 to allow for more explanatory variable names. It is important to have a guideline for how it should be written and formatted in large projects, especially when a lot of people will be working with it. The guideline serves to standardise the code, such that it is easier to process and understand when new members of the project look at the code (or you return after a year of not looking at it). Furthermore, standardised code can also entail using comments in a certain way, i.e. to explain why something is implemented the way it is.


## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement and what are they testing in your code?**
>
> Answer length: 50-100 words.
>
> Example:
> *In total we have implemented X tests. Primarily we are testing ... and ... as these the most critical parts of our*
> *application but also ... .*
>
> Answer:

We implemented 2 main calls containing different tests for data and model, respectively. In total we implemented three tests for testing the training, test and validation data. They check that the data-set is of correct length and shape, and that each sample has a corresponding label. Furthermore, we ensure that all classes are present.

The model tests checks that the shape of the input data is correct. To account for this two tests have been included, one to check that it is a 4D tensor that is given, and one that ensures that each image input has size `[batch_size, 3, 224, 224]`.



### Question 8

> **What is the total code coverage (in percentage) of your code? If you code had an code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> Answer length: 100-200 words.
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our **
> *code and even if we were then...*
>
> Answer:

Coverage is calculated on the two Pytest test scripts. The total code coverage of the two scripts is 100%, which includes tests of model and data. Since the code-coverage is so high we would expect the code to have fewer errors. However, this could lead to a false sense of security since it is not guaranteed error-free. Wrong arguments can be passed between functions which work fine on their own but fail under specific conditions. All the possible inputs and edge cases might not be tested and such inputs might lead to the program failing or, even worse, give a wrong result which is unknown to us, and thereby skewing the results. Code coverage only refers to the percentage of our code that actually gets run when all our tests are executed.

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Answer length: 100-200 words.
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in*
> *addition to the main branch. To merge code we ...*
>
> Answer:

Yes we used both branches and pull requests in our workflow. Branches were created whenever a new feature were added to the code, such that there were a specific branch for that feature. When it was fully developed it was made into a pull request where another group member closely examined the code and solved the merge conflicts. If there were any conflicts that the reviewing group member thought were problematic, they talked about the conflict and found a solution. This ensured that we had a good overview of our code changes and were able to experiment without interfering with the main code. 

### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
> Answer:

Yes DVC was used to manage the data. Since the data used in this project is 224x224 images and there are approximately 20.000 of them, the resulting data files take up several Gbs of space. Therefore, sharing the data between members of the group and controlling that it is consistent for all members would be extremely cumbersome without DVC. With DVC retrieving the data was a matter of writing `dvc pull` and waiting a few minutes. It also is essential for ensuring reproducibility. If a new user joins the project, the user can be certain that the data retrieved is exactly as it should be.

### Question 11

> **Discuss you continues integration setup. What kind of CI are you running (unittesting, linting, etc.)? Do you test**
> **multiple operating systems, python version etc. Do you make use of caching? Feel free to insert a link to one of**
> **your github actions workflow.**
>
> Answer length: 200-300 words.
>
> Example:
> *We have organized our CI into 3 separate files: one for doing ..., one for running ... testing and one for running*
> *... . In particular for our ..., we used ... .An example of a triggered workflow can be seen here: <weblink>*
>
> Answer:

We use unit testing to cover base cases of both model and data. The tests are split into four files: One for the tests themselves, one for checking compliance with pep8 via flake8, one to run isort and finally one to check the coverage of our code. The unit tests are set up to trigger via github actions on every pull request. Furthermore, we made use of the git tag function to set up automatic docker building in the cloud on google cloud platform, such that when a pull request was made, if it included the tag ???cloud-build???, it builds new images on gcp. The tests and coverage were run on ubuntu, macos and windows, as well as python 3.8 and 3.9, to ensure compatibility across various installations. The isort and flake8 tests were only conducted on a single python version and os, as they are simply checks for compliance with good code practices. We make use of caching for the tests and code coverage as they require a great amount of setup for testing across multiple python versions and operating systems. An example of a triggered workflow can be seen [here](https://github.com/rreezN/dtu_mlopsproject/actions/runs/3929433014).

## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: python my_script.py --lr 1e-3 --batch_size 25*
>
> Answer:

We ensured reproducibility by using hydra to load hyperparameters from a `config/` folder containing hyperparameters for the model and training in separate config files within subfolders. Moreover, all hyperparameters of each run are stored using WandB, meaning that the hyperparameters of each experiment are stored and can not accidentally be overwritten. This means that when the config-files are filled in, the model can be trained using: 

```bash
python src/models/train_model.py
```

### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:

As written above, we used a combination hydra and WandB to ensure that all information regarding the experiments was saved along with the trained model. This means that hydra loaded the hyperparameters set within the separate config-files for the model and training setup. During training everything within the config files is stored by WandB, meaning that the hyperparameters corresponding to a particular experiment are not accidentally overwritten. To further ensure reproducibility, the training and prediction process of course starts by setting a seed. This has the effect that initialising weights and shuffling data inside the dataloaders becomes deterministic rather than stochastic in the sense that the same ???random??? outcome happens each time.

### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:

![](figures/wandb1.png)

First and foremost we decided to log graphs of the training/validation loss and training/validation accuracy, as these four metrics give us the most immediate idea of how a given model is performing. The above figure shows exactly these metrics and how the different models  compare against each other.

![](figures/wandb2.png)

In addition to model performance metrics, we also log all the relevant model and training information for a given run. In the above figure we see that this information includes the batch size, number of epochs, learning rate, model name etc. From the left side of the figure, we also see that we have several models saved for a single run. This is done using Pytorch Lightning???s callback function ModelCheckpoint that saves a model each time the validation loss reaches a new low within a given run.

![](figures/wandb3.png)

The final thing we decided to track is the confusion matrix for the model in each epoch, as is seen above. The confusion matrix shows us the model???s predicted classes vs. the actual classes and thus gives us an indication of which animals the model struggles on during the training process.
We did not decide to do any hyperparameter sweeping this time around because of different issues regarding Hydra and Wandb.


### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments? Include how you would run your docker images and include a link to one of your docker files.**
>
> Answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
> Answer:

Two containers, trainer and deployment, were created based on two image builds. We did not utilise local dockers, however, we did use google???s cloud containers. Link to our container registry has trainer images and deployment images (project_app) can be found [here]( https://console.cloud.google.com/gcr/images/eternal-seeker-374308?project=eternal-seeker-374308)
```bash
gcloud ai custom-jobs create --region=europe-west1 --display-name=test-runXX --config=config_cpu.yaml
```
Where config_cpu.yaml is defined as 
```yaml
workerPoolSpecs:
   machineSpec:
      machineType: c2-standard-8
   replicaCount: 1
   containerSpec:
      imageUri: gcr.io/eternal-seeker-374308/trainer
```
We specifically use the c2 machine type because it offers great computing power.


### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer:

Most debugging was done through the different IDEs which the group members used. These are in particular Pycharm and Visual Studio Code. This way it is simple to set up a debug configuration, to go through the code line-by-line when encountering errors in the code. Later on, when everything seemed to run, errors arose when building docker images and in particular when attempting to train in the cloud. Here debugging was much more difficult since you were only left with a single error message and re-running takes more and more time the further along in the process you have come. We tried profiling the code, but since we used pytorch-lightning, it was very difficult to see how we ourselves were to improve performance noticeably, not because the code is perfect by any means, but because most computations are handled elsewhere.

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer:

To some extent we used all the following services during the development of this project:
* Buckets: 
    * Makes it possible to store data and trained models remotely with easy access.
* Compute Engine:       
    * Virtual machine computing capabilities with GPU support.
* Vertex AI:            
    * Easily build, deploy, and manage AI and machine learning development.
* Container Registry:   
    * Private Docker image storage system
* Cloud Build:
    * Executes one???s builds on Google Cloud Platform infrastructure by importing source code from github.
* Cloud Function:
    * Set up cloud deployment, such that we can predict on images.
* Cloud Run:
    * Set up cloud deployment via docker images and fast api


### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Answer length: 100-200 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:

Initially we wanted to use Compute Engine for model training, however, due to some billing issues and troubles regarding deployment of docker images we were unable to use the service. Furthermore, because Vertex AI is offered as another service, which, additionally, fits our needs better we chose to train with Vertex AI instead of VM instances. With Vertex AI automatically creating a VM based on a container image that it has grabbed from our Container Registry, executing the training and shutting down the VM again, this service just fits us better.  As for hardware specifications we configured a .yaml file for CPU training which is seen below. As we ran into multiple problems regarding getting Vertex AI to create an instance with a GPU accelerator, we decided on the powerful ???c2??? CPU machine type to speed up training.

```yaml
workerPoolSpecs:
   machineSpec:
      machineType: c2-standard-8
   replicaCount: 1
   containerSpec:
      imageUri: gcr.io/eternal-seeker-374308/trainer
```


### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

![](figures/bucketlist.png)

### Question 20

> **Upload one image of your GCP container registry, such that we can see the different images that you have stored.**
> **You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:

![](figures/container-registry.png)

### Question 21

> **Upload one image of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:

![](figures/cloud-build.png)

### Question 22

> **Did you manage to deploy your model, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:

Initially, we deployed the model locally using torchserve. The after some trial-and-error with deploying in the cloud, we managed to deploy our model both using Cloud Functions and Cloud Run. Since the model requires a lot of dependencies, such as torch, torchvision, pytorch-lightning etc., we ended up using Cloud Run to deploy our model with an image dedicated to predictions. The service is build using fastapi which simplified sending and reading image files through requests. To invoke the service a user just needs to call:
```bash
curl -X 'POST' \
  'https://project-app-gqbczfp77a-ew.a.run.app/' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@tiger.jpg;type=image/jpeg'
```
Where tiger.jpg is replaced with whatever image the user wants the model to predict on.


### Question 23

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:

We did not implement monitoring of your deployed model. Monitoring would help the longevity of our application by keeping an eye on some key aspects. The performance is monitored to detect if there are slow response times and or errors. This will help us find issues with the model, which then can be taken care of. Furthermore, by looking at the performance, it is possible to detect if there is a drift in the data in any way, which is likely to happen since new pictures would have other resolutions. The resources may also be monitored to ensure that the capacity is not reached regarding latency and throughput such that the model should be optimized. 

### Question 24

> **How many credits did you end up using during the project and what service was most expensive?**
>
> Answer length: 25-100 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ...*
>
> Answer:

During the project, an alleged bug occurred where all our student billing accounts got emptied overnight (no VM instances were active and no spend history). Therefore, two of the group members created trial accounts with 300$ each of which around 60$ was spent. 

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 25

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally in your own words, explain the**
> **overall steps in figure.**
>
> Answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and puch to github, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:

The Machine Learning Operations Pipeline is split into two sides: The developer and the user.
The development environment is where the model is being developed. This environment points to a Local data storage where the data used for training the model is stored. The Local data storage points to a Data version control which is used to keep track of different versions of the data and manage the data pipeline. The Data version control points to a Bucket, which is used to store the data in the cloud. The bucket is connected to the Cloud deployment, where the model is deployed to a cloud-based server.

The development environment also points to GitHub, a web-based platform used for version control and collaborative development. GitHub points to the data version control, which is used to keep track of different versions of the model code and manage the development pipeline.
GitHub also points to GitHub Actions, which is a feature that allows users to automate tasks such as building, testing, and deploying code.

GitHub also points to a Container Registry, which stores and distributes containers, such as Docker images. The Container Registry points to Vertex AI, a platform for managing machine learning models, and it is connected to both Bucket and WandB for data storage and monitoring.
GitHub also points to Docker, a platform for building, shipping, and running distributed applications in containers.

Finally, there is a User who is interacting with the model, this user points to the Server, Docker and GitHub. This means that the user can interact with the deployed model on the server as well as using the Docker container and clone the repository for local running through the GitHub platform.

### Question 26

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:

By far, the biggest struggles of the project were building the docker images and integrating the model into google cloud platforming. We spend a disproportionate amount of time getting everything set up to run in the cloud compared to getting the model running locally, setting up CI, ensuring reproducibility etc.
It turns out that building docker images through wsl uses A LOT of memory. This slowed the process of building and pushing images down in the beginning, but when we figured out how to limit the amount of ram wsl has access to, this problem was solved.
The most significant issue with the cloud integration was trying to enable training on GPU. We never actually managed to get it up and running, mostly due to the fact that the 50$ credits expired over night and free accounts have limited access to GPU-machines. Training on CPU worked fine, however, it meant that we had to significantly reduce the amount of data we trained on.
Another general challenge was the lack of transparency and understandability of the underlying logic behind downloading and configuring our machines to work with third-party software. The solution was, of course, to google the error and then find suitable answers written by wizards, but in general, it was unclear why a solution would fix an error, and this black box working often led to some confusion and frustration. It sometimes felt like fumbling forward blindly and hoping we did not reach a dead end. 


### Question 27

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project**
>
> Answer length: 50-200 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
>
> Answer:

Student s194237 was mainly in charge of??? Development environment??? and ???Debugging, Profiling and Logging???
Student s194246 was mainly in charge of ???The cloud???
Student s194247 was mainly in charge of ???Organization and version control??? and ???Reproducibility???
Student s194249 was mainly in charge of ???Deployment???
Student s194267 was mainly in charge of ???Continuous Integration???

All members contributed to code by helping each other out and solving issues together when they arose.

