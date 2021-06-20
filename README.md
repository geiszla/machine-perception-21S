# 3D Pose Prediction

Project Link: https://machine-perception.ait.ethz.ch/project4/

## How to reproduce the results

cd to the root folder and

Run:

```bash
python -u src/train.py --model [model] --n_epochs 2000 --lr 0.0005 --seed 42 --divide_lr_every 400
```

where [model] can be either:

* `ConvAttModel` for the GCN with the convolutional 1d network
* `GCNAttModel` for the GCN model with encoder/decoder structure,
* `GCNModel` for the GNC model.

## Development workspace and configuration

### Create new models

To create a new model, create a new python script in the `models` folder. This script has to define a class which inherits from the `BaseModel` class found in `base_model.py`. You can then select to run `train.py` with this newly defined model using the tag `--model` followed by the name of the class. If no model arguments are given, the `DummyModel` will be selected by default.
The `create_model` function in the models module will go through all the classes in all the scripts in the `models` folder, so you could technically define two different model classes in one script if necessary.

### Testing

Visualization of the predictions can be done using `evaluation.py`. If ran with only `--model_id` provided, it will predict the target sequence of the test data and display 10 randomly picked samples. If `--eval_on_val` is provided and its value is `1`, it will evaluate the model on the validation set and display the prediction alongside the ground truth.

### Formatting

The predictions on the test data are in a `.csv.gz` format which can be directly uploaded on the submission website. After training, this file will be automatically generated and put in the model folder in your experiment folder along with the saved model parameters and configuration.

### How to run locally

#### Using conda

1. Create a virtual environment in conda

   ```bash
   conda create --name MP python=3.7.4
   ```

1. Activate it

   ```bash
   conda activate MP
   ```

1. Install `pytorch` and `torchvision`

   ```bash
   conda install pytorch=1.6.0 torchvision=0.7.0
   ```

1. Then, install requirements. You might have to comment out torch and torchvision.

   ```bash
   conda install --file requirements.txt
   ```

1. Add MP_DATA and MP_EXPERIMENTS environment variables. This is how to do it on windows:

   ```bash
   conda env config vars set MP_DATA=..\..\project4_data
   conda env config vars set MP_EXPERIMENTS=..\experiments
   ```

1. It should now run

   ```bash
   python train.py --model DummyModel
   ```

### How to run on Leonhard

1. Connect to the Leonhard host (with your terminal or with VS Code) with

   ```bash
   ssh [ethzusername]@login.leonhard.ethz.ch
   ```

1. Clone the project (only for the first time)

   ```bash
   git clone [ssh project link from Gitlab] 
   ```

   *Warning: it could be necessary to copy the ssh key from Leonhard to Gitlab to have access to the Repository*

1. Install the Python GPU Module on Leonhard ([see wiki](https://scicomp.ethz.ch/wiki/Getting_started_with_clusters)) with:

   ```bash
   module load python_gpu/3.7.4
   ```

1. Run

   ```bash
   export MP_DATA="/cluster/project/infk/hilliges/lectures/mp21/project4"
   ```

   and

   ```bash
   export MP_EXPERIMENTS="$HOME/Appliedscience/experiments"
   ```

1. Add these commands to the bashrc file to automatically run them when starting up the cluster:

   ```bash
   module load python_gpu/3.7.4
   source $HOME/.local/bin/virtualenvwrapper.sh
   workon "MP21"
   export MP_DATA="/cluster/project/infk/hilliges/lectures/mp21/project4"
   export MP_EXPERIMENTS="$HOME/Appliedscience/experiments"
   ```

### Running the script on leonhard

Go to the folder where your `train.py` file is located. You can run the following command:

```bash
bsub -n 1 -W 4:00 -o [outputname] -J [jobname] -R "rusage[mem=8096, ngpus_excl_p=1]" python -u train.py --model [model_class_name]
```

You can add other CL arguments like --lr or --n_epochs.
*Careful*: the output file of the job will be saved in the folder you submitted the job from. This means that it will end up in the local git repo on your Leonhard home if you run the above command. Please do not push anything that contains a job output file onto the github as this would make it messy very quickly. Either have some sort of prefix system and add all files that start with said suffix to .gitignore or you can submit the job from outside of your git repo :

```bash
bsub -n 1 -W 4:00 -o [outputname] -J [jobname] -R "rusage[mem=8096, ngpus_excl_p=1]" python -u mp_project/src/train.py --model [model_class_name]
```
