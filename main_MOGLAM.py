
from train_test import train_test


if __name__ == "__main__":

    data_folder = './BRCA_split/BRCA'

    view_list = [1,2,3]
    num_epoch_pretrain = 500
    num_epoch = 3000

    theta_smooth = 1
    theta_degree = 0.5
    theta_sparsity = 0.5


    if data_folder == './BRCA_split/BRCA':
        num_class = 5
        lr_e_pretrain = 1e-4
        lr_e = 1e-5
        lr_c = 1e-6
        reg = 0.001
        neta = 0.1


    train_test(data_folder, view_list, num_class,
               lr_e_pretrain, lr_e, lr_c, 
               num_epoch_pretrain, num_epoch, theta_smooth, theta_degree, theta_sparsity, neta, reg)