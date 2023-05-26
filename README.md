# MOGLAM
With the rapid development and accumulation of high-throughput sequencing technology and omics datasets, many studies have conducted a more comprehensive understanding of human diseases from a multi-omics perspective. Meanwhile, the graph-based method has been widely used to process multi-omics data due to its powerful expressive ability. However, most existing graph-based methods utilize fixed graphs to learn sample embedding representations, which often leads to sub-optimal results. Furthermore, treating embedding representations of different omics equally usually cannot obtain more reasonable integrated information. In addition, the complex correlation between omics is not fully taken into account. To this end, we propose an end-to-end interpretable multi-omics integration method, named MOGLAM, for disease classification prediction. Dynamic graph convolutional network with feature selection is first utilized to obtain higher quality omics-specific embedding information by adaptively learning the graph structure and discover important biomarkers. Then, multi-omics attention mechanism is applied to adaptively weight the embedding representations of different omics, thereby obtaining more reasonable integrated information. Finally, we propose omics-integrated representation learning to capture complex common and complementary information between omics while performing multi-omics integration. Experimental results on three datasets show that MOGLAM achieves superior performance than other state-of-the-art multi-omics integration methods. Moreover, MOGLAM can identify important biomarkers from different omics data types in an end-to-end manner.
# The workflow of MOGLAM method
![The workflow of MOGLAM method](https://github.com/Ouyang-Dong/MOGLAM/blob/master/workflow.jpg)
# Introduction to code
The repository mainly includes the following datasets and *.py* files as shown below：
1. BRCA dataset for breast invasive carcinoma PAM50 subtype classification.
2. The detailed *.py* files introduction are as follows:

    2.1 *main_MOGLAM.py* : This is the main function, we only run it to train the model, which can output the prediction performance on the test set, namely ACC, F1_weighted and F1_macro.
    
    2.2 *train_test.py* : In the *train_test.py* file, we define the `prepare_trte_data` function for reading datasets, the `gen_trte_adj_mat` function for calculating the patient similarity matrix, the `train_epoch` function for training the model and the `test_epoch` function for testing the model.
    
    2.3 *models.py* : In the *models.py* file, we define the `GraphLearn` class for adaptive graph learning, the `GCN_E` class for graph convolutional network learning, the `Multiomics_Attention_mechanism` class for multi-omics attention learning and the `TransformerEncoder` class for omic-integrated representation learning.
    
    2.4 *layers.py* : In the *layers.py* file, we mainly define the `Attention` class for self-attention learning, the `FeedForwardLayer` class for feedforward network learning, and make use of the `EncodeLayer` class to build query matrix, key matrix, value matrix and multi-head self-attention layers.
    
    2.5 *utils.py* : In the *utils.py* file, we mainly define the `cosine_distance_torch` function for cosine similarity calculation, the `gen_adj_mat_tensor` function for patient similarity matrix construction and the `GraphConstructLoss` function for adaptive graph structure loss learning.
    
    2.6 *param.py* : In the *param.py* file, we define the `parameter_parser` function for setting hyperparameters.

# How to run the code
Although we build several *.py* files, running our code is very simple. More specifically, we only need to run *main_MOGLAM.py* to train the model, outputting prediction results. In addition, running our code requires utilizing PyTorch's deep learning framework under Python 3.8.

# Environment Requirement
The code has been tested running under Python 3.8. The required packages are as follows:
- torch == 1.12.1 (GPU version)
- numpy == 1.23.5
- pandas == 1.5.0
# Tutorial
For the step-by-step tutorial and a detailed introduction to defined classes and functions, please refer to: [https://moglam.readthedocs.io/en/latest/](https://moglam.readthedocs.io/en/latest/)
