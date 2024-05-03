# Scripts for generating SHAP & LIME explainations and their corresponding plots

### Usage:
To use either the SHAP or LIME scripts, first ensure that you have both the `UNSW_NB15_testing-set.csv` and `UNSW_NB15_training-set.csv`CSV files in your working directory. (Note: You may need to adjust the paths in the scripts to fit your device). 
Then run the file based on what model need. For SHAP, the naming convention is: `UNSW_SHAP_[Model].py`. For LIME, the naming convention is: `LIME_[Model],ipynb`.

Time taken for training heavily depends on how much of the dataset is used in the Dataframe, defined by `frac= ` parameter within the `df = df.sample()` function. Using a larger fraction of the dataset can lead to undesirable behaviors, including system freezing, inaccurate outputs, and out-of-memory errors. 
To successfully process the entire dataset, ensure that your system is equipped with at least 64GB of RAM. Attempting to run the code with less memory **will** result in an out-of-memory error.

