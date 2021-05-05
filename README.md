# 3D Pose Prediction

Project Link: https://machine-perception.ait.ethz.ch/project4/ 

## Introduction [TODO]:
1. Poetry
2. Project code structure
3. Development workspace and configuration
4. Testing
5. Formatting

## How to run locally [TODO]

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