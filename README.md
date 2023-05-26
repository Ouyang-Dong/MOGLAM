# MOGLAM
With the rapid development and accumulation of high-throughput sequencing technology and omics datasets, many studies have conducted a more comprehensive understanding of human diseases from a multi-omics perspective. Meanwhile, the graph-based method has been widely used to process multi-omics data due to its powerful expressive ability. However, most existing graph-based methods utilize fixed graphs to learn sample embedding representations, which often leads to sub-optimal results. Furthermore, treating embedding representations of different omics equally usually cannot obtain more reasonable integrated information. In addition, the complex correlation between omics is not fully taken into account. To this end, we propose an end-to-end interpretable multi-omics integration method, named MOGLAM, for disease classification prediction. Dynamic graph convolutional network with feature selection is first utilized to obtain higher quality omics-specific embedding information by adaptively learning the graph structure and discover important biomarkers. Then, multi-omics attention mechanism is applied to adaptively weight the embedding representations of different omics, thereby obtaining more reasonable integrated information. Finally, we propose omics-integrated representation learning to capture complex common and complementary information between omics while performing multi-omics integration. Experimental results on three datasets show that MOGLAM achieves superior performance than other state-of-the-art multi-omics integration methods. Moreover, MOGLAM can identify important biomarkers from different omics data types in an end-to-end manner.
# The workflow of MOGLAM method
![The workflow of MOGLAM method](https://github.com/Ouyang-Dong/MOGLAM/blob/master/workflow.jpg)
# Introduction to code
The repository mainly includes the following datasets and *.py* files as shown below：
1. BRCA dataset for breast invasive carcinoma PAM50 subtype classification.
2. The detailed *.py* file introduction is as follows:

    2.1 *main_MOGLAM.py*: 
    
    2.2 *train_test.py*:
    
    2.3 *models.py*:
    
    2.4 *layers.py*:
    
    2.5 *utils.py*:
    
    12.6 *param.py*:
# Environment Requirement
The code has been tested running under Python 3.8. The required packages are as follows:
1. torch == 1.12.1 (GPU version)
2. numpy == 1.23.5
3. pandas == 1.5.0
# Tutorial
For the step-by-step tutorial and a detailed introduction to defined classes and functions, please refer to: [https://moglam.readthedocs.io/en/latest/](https://moglam.readthedocs.io/en/latest/)
