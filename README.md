# MGpi:  A Computational Model of Multiagent Group Perception and Interaction

## Synthetic Dataset (Group Layouts)
https://www.dropbox.com/s/e1rzcbeeu86eh96/interaction_data.zip?dl=0

## Dependency Installations
This installs dependencies in a virtual environment, inheriting system packages. 
Run ```chmod u+x setup.sh;  ./setup.sh```

## Simulating Communication Interactions
This will generate training data for our MGpi Network
1. Download and extract the Synthetic Dataset linked above 
2. Run ```source virt_mgpi/bin/activate```. This activates the virtual environment with dependencies installed.
3. Run ```python simlulate.py```. This imports GroupInteract from sim\_interact.py or sim\_interact\_dyn.py, and simulates group communications.

## Training policy
1. After you have simulations, chane appropriate variables in train.py and run ```nohup python train.py simulations_directory model_name number_of_epochs```. 
2. model_name here can be:

AllHist for MGpi

All-NoSelfState for baseline NSO

NoNeighMod for baseline SSO

All-NoKPMGate for baseline EQPOOL

All-SocPool for baseline SOCPOOL

## Testing policy
After changing appropriate variables in test.py, run ```nohup python test.py```

## Testing performance on group detection
After changing appropriate variables in test\_group.py, run ```nohup python test_group.py```
