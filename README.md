This repository is dedicated to a table data filling algorithm ‘DDGANI’, offering several UCI dataset examples and corresponding environment setup.

Our setup environment is notably simple, detailed as follows:

**## Setup Environment**

*```*

conda create --name DDGANI --yes

conda activate DDGANI 

conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

conda install pandas
conda install scikit-learn 
conda install logging argparse tqdm

*```*



Algorithm parameters can be adjusted in the '/param/param.json' file, where you can also include parameters for your unique datasets. The key configurations are explained below:

**## Configurations**

"name": Dataset name
"T": Steps for diffusion
"file_path": Dataset location
"categorical_cols": Collection of categorical columns
"top_k": Top k data for attention-based filling
"model_name":"minmax", Method for preprocessing numerical data
"loss_weight": Weight of the loss

To run the code, execute the following command:

**## Run an Experiment**

*```*

cd test_main

python main.py --Data adult --MissType MCAR --MissRate 0.2 --UseAttention True --UseLearner True --UseLearner True

*```*

The 'Data' parameter must match the 'name' in the '/param/param.json' file. 'MissType' specifies the type of missing data, options include 'MCAR', 'MNAR', 'MAR', and 'Region'. 'MissRate' sets the missing rate.
You can also enable attention mechanisms, learners, and data dependency plugins by setting UseAttention, UseLearner, and UseFD respectively.

For adding new datasets, follow these steps:

**## Adding a Tabular Dataset**

*```*

Add datasets under the 'dataset/mix datasets or numerical datasets' folder and configure the parameters in '/param/param.json'.

*```*

Moreover, we provide various publicly available or replicated data filling methods in the 'BaseLine' folder.
Most methods can be executed directly by modifying the code in the '/test_main/main.py' file.
The remaining methods have also been open-sourced, and their execution procedures can be viewed on GitHub at your convenience.