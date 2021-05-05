# 3D Pose Prediction

Project Link: https://machine-perception.ait.ethz.ch/project4/ 

## Introduction [TODO]:
1. Poetry
2. Project code structure
3. Development workspace and configuration

# Create new models
To create a new model, create a new python script in the "models" folder. This script has to define a class which inherits from the 'BaseModel' class found in base_model.py. You can then select to run train.py with this newly defined model using the tag '--model' followed by the name of the class. If no model arguments are given, the DummyModel will be selected by default.
The create_model function in the models module will go through all the classes in all the scripts in the "models" folder, so you could technically define two different model classes in one script if necessary.

4. Testing
5. Formatting

## How to run locally [TODO]
# Using conda
Create a virtual environment in conda : 

```
conda create --name MP python=3.7.4
```

Activate it:
```
conda activate MP
```

Install torch and torchvision :

```
conda install torch=1.6.0 torchvision=0.7.0
```

Then, install requirements. You might have to comment out torch and torchvision.

```
conda install --file requirements.txt
```

Add MP_DATA and MP_EXPERIMENTS environment variables. This is how to do it on windows : 

```
conda env config vars set MP_DATA=..\..\project4_data
conda env config vars set MP_EXPERIMENTS=..\experiments
```

It should now run:

```
python train.py --model DummyModel
```

## How to run on Leonhard

Connect to the Leonhard host (with your terminal or with VS Code) with 

```
ssh [ethzusername]@login.leonhard.ethz.ch
```

(only for the first time) clone the project with:

``` 
git clone [ssh project link from Gitlab] 
```

(Warning: it could be necessary to copy the ssh key from Leonhard to Gitlab to have access to the Repository)

Install the Python GPU Module on Leonhard (https://scicomp.ethz.ch/wiki/Getting_started_with_clusters) with :

```
module load python_gpu/3.7.4
```

run 
```
export MP_DATA="/cluster/project/infk/hilliges/lectures/mp21/project4"
```

and 

```
export MP_EXPERIMENTS="$HOME/Appliedscience/experiments"
```

and add these commands to the bashrc file to automatically run them when starting up the cluster:

```
module load python_gpu/3.7.4
source $HOME/.local/bin/virtualenvwrapper.sh
workon "MP21"
export MP_DATA="/cluster/project/infk/hilliges/lectures/mp21/project4"
export MP_EXPERIMENTS="$HOME/Appliedscience/experiments"
```