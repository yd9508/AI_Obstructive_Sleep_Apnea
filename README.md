The running time comparison results are summarized in the file Report for FedL.docx.
To run the code locally, please run the following commands in Command Prompt.
```
git clone https://github.com/AI-Health-Lab/Ray-FL-FDA/
cd Ray-FL-FDA
python main.py
```
Please note that you need to specify which federated learning frameworks you want to test. Please input either "HFL" or "VFL". 
You can also check the running results in the notebook file Simulation_Ray_FL_FDA.ipynb.

As Tables 1 and 2 show, the tests perform simulations to compare the running time with the help of a distributed parallel computing framework, Ray. We reported the average running time of 5 replications.
![alt text](https://github.com/AI-Health-Lab/Ray-FL-FDA/blob/main/img/testResults.png?raw=true)
