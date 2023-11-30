To run this project, you may need to set up the following environment:
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install pandas
conda install scikit-learn

This project is designed for data imputation. We've demonstrated its capabilities using two examples from the UCI repository.
You can execute the program by running `main.py` under the `test_main` directory; 
the program will output the ARMSE, AMAE of the imputed data, as well as its accuracy on downstream tasks.
Additionally, you can customize the parameters in `param.json` within the `param` directory and add new datasets based on your requirements.

You can execute the code and selectively activate certain plugins using the following command:
python main.py --Data adult --MissType MCAR --MissRate 0.2 --UseAttention True --UseLearner True --UseFD True
If you want to enable/disable a plugin, you only need to make the param False，like --UseFD False
