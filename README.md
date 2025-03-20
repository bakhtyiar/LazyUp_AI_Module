# Get started
1. Install anaconda3. I have 2024.10.1 version and python 3.12.x within
2. run in anaconda prompt cli in project root directory 
   1. `conda env create -f environment.yml`
   2. `conda activate myenv`
3. Now you are able to run scripts properly

---

## Queue of actions

1. Create data
   1. run "generate..." py scripts to create mock data, 
   2. or run "listen..." py scripts to collect data, 
   3. also you might run listeners over gui interface in "gui" folder with controls to regulate target value for future training model
2. Train model
   1. run "train..." py scripts to train models and save persistent data across reuses
3. Try prediction
   1. run "predict..." py scripts to test prediction efficiency of models
4. Try everyday app form
   1. run "main_menu" py script
   2. run from this gui required feature