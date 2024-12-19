![image](https://github.com/user-attachments/assets/832b5112-a022-48fe-8f58-d308553af342)# sc-blocks-mining

# RPA PA RA
The startup function for RA, PA, and RPA is "Converge.main". It requires seven arguments to be input as follows: "InputPath OutputPath delimeter k pro1 pro2"  
**InputPath**: File path for input data  
**OutputPath**: Path for result output  
**delimeter**: delimeter  
**k**: Number of spatiotemporal blocks I hope to find  
**pro1,pro2**: The proportion of density and space in score function f. The proportion of pro1 in f is pro1/(pro1 + pro2).
We have provided a test data in data folder. you can run the test data by entering the parameters "data/test.txt output/test "," 10 1 1" in main function".  

# EA
The startup function for EA is "Converge.main". It requires seven arguments to be input as follows: "InputPath OutputPath delimeter k h pro1 pro2"  
**InputPath**: File path for input data  
**OutputPath**: Path for result output  
**delimeter**: delimeter  
**k**: Number of spatiotemporal blocks I hope to find  
**h**: How many intervals are the spatiotemporal range divided into.  
**pro1,pro2**: The proportion of density and space in score function f. The proportion of pro1 in f is pro1/(pro1 + pro2).  
We have provided a test data in data folder. you can run the test data by entering the parameters "data/test.txt output/test "," 10 100 1 1" in main function.  

# User Manual
In the user manual, we only explain how to execute the algorithm demo using IDEA. In addition to that, users can also package the code into a JAR file and run it on the local command-line console on Windows or iOS systems.
## 1、Code Clone Or Download
To download the code to your local folder, you can use the following command in your terminal or command prompt:
'''
git clone https://github.com/tangweike778/SP_Block.git
'''
Additionally, the code project can also be obtained by downloading and extracting the zip file.
## 2、Opening the Code Project Using IDEA
![image](https://github.com/user-attachments/assets/7665216b-7380-43c5-9dfc-d4cbd9d28a5e)
## 3、Open the algorithm file you want to run and locate the main file named "Converge," which contains the main function to start the execution.
![image](https://github.com/user-attachments/assets/1af9caf3-f099-4343-bd4e-2fddb8eab47a)
## 4、Configure the startup parameters based on the corresponding algorithm's parameter description mentioned above.
### 4.1、Click the Run button and select the "Modify Run Configuration" option.
![image](https://github.com/user-attachments/assets/bf315924-9068-455f-8b39-8dadc8f1dd05)
### 4.2、As shown in Step 1 of the figure below, first modify the working directory to the directory of the code file to be executed.
### 4.3、As shown in Step 2 of the figure below, fill in the algorithm parameters according to the parameter description provided above.
![image](https://github.com/user-attachments/assets/0982345d-1993-4776-b601-e9042034febc)
## 5、Click the Run button to execute the specified algorithm.
![image](https://github.com/user-attachments/assets/daec8c63-1c7a-4e92-9a5a-014322dc035b)
## 6、Wait for the algorithm to complete its execution and obtain the basic results along with the corresponding output files.
![image](https://github.com/user-attachments/assets/81d05c15-0c4f-48ec-a807-3b0bb860b181)
