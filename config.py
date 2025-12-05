import argparse

parser = argparse.ArgumentParser()


"""
For Traditional Federated Learning
"""
# basic settings
parser.add_argument("--mode", type=str, default='debug', choices=['debug', 'local', 'online'])
parser.add_argument("--seed", type=int, default=2024)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--device_id", type=str, default="0")

# dataset settings
parser.add_argument("--dataset_root", type=str, default="./datasets/data")
parser.add_argument("--split_file", type=str, default="")
parser.add_argument("--dataset", type=str, default='CIFAR10')
parser.add_argument("--dir_alpha", type=float, default=0.5)

# fl global settings
parser.add_argument("--num_clients", type=int, default=10)
parser.add_argument("--model", type=str, default="ConvNet", choices=['ConvNet', 'ConvNetBN', 'ResNet18', 'ResNet18BN'])
parser.add_argument("--communication_rounds", type=int, default=300)
parser.add_argument("--join_ratio", type=float, default=0.2)

parser.add_argument("--eval_gap", type=int, default=1)
parser.add_argument("--algorithm", type=str, default="FedTC",
                    choices=['FedAvg','FedTC', 'FedAvgM', 'FedProx','SCAFFOLD', 'MOON', 'FedDyn', 'FedGen', 'FedLC', 'FedDM', 'FedAF', 'FedPAD'])
parser.add_argument("--local_epochs", type=int, default=5)
parser.add_argument("--local_batch_size", type=int, default=32)
parser.add_argument("--local_learning_rate", type=float, default=0.01)
parser.add_argument("--local_momentum", type=float, default=0.0)
parser.add_argument("--local_weight_decay", type=float, default=0.0)

# FedProx/MOON settings
parser.add_argument("--mu", type=float, default=0.01)
# SCAFFOLD settings
parser.add_argument("--server_learning_rate", type=float, default=1.0)
# FedDyn settings
parser.add_argument("--alpha", type=float, default=0.01)
# MOON/FedLC settings
parser.add_argument("--tau", type=float, default=0.5)
# FedGen settings
parser.add_argument('-nd', "--noise_dim", type=int, default=512)
parser.add_argument('-glr', "--generator_learning_rate", type=float, default=0.005)
parser.add_argument('-hd', "--hidden_dim", type=int, default=512)
parser.add_argument('-se', "--server_epochs", type=int, default=1000)
parser.add_argument('-lf', "--localize_feature_extractor", type=bool, default=False)
parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99) # for generative model
# FedAvgM settings
parser.add_argument('-bt', "--beta", type=float, default=0.9)

"""
For Aggregation Free Federated Learning
"""
parser.add_argument("--global_model_epochs", type=int, default=500, help="global model training epochs")
parser.add_argument("--global_batch_size", type=int, default=256, help="global model batch size")
parser.add_argument("--global_learning_rate", type=float, default=0.01)
parser.add_argument("--global_momentum", type=float, default=0.9)
parser.add_argument("--global_weight_decay", type=float, default=0.0)
parser.add_argument("--global_learning_rate_decay", action="store_true", default=True)
# FedDM settings
parser.add_argument("--ipc", type=int, default=50)
parser.add_argument("--rho", type=int, default=5, help='rho-radius ball for gradient clipping')
parser.add_argument("--gnb", type=int, default=5, help='gradient normal bound for the Gaussian noise')
parser.add_argument("--sigma", type=int, default=0, help='sigma for the Gaussian noise')
parser.add_argument("--dc_iterations", type=int, default=1000)
parser.add_argument("--dc_batch_size", type=int, default=256, help='real image batch size for data condensation')
parser.add_argument("--image_lr", type=float, default=1.0)
parser.add_argument('--init', type=str, default='real', choices=['noise', 'real', 'mix'],
                    help='initialization of synthetic data, noise/real/mix: initialize from random noise or real or mix images.')
parser.add_argument("--single", action='store_true', help='use single round syn images')

# dsa
parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                    help='differentiable Siamese augmentation strategy')

# FedAF
parser.add_argument("--lamda_loc", type=float, default=0.0001)
parser.add_argument("--lamda_glob", type=float, default=0.01)

# FedPAD
parser.add_argument("--balance", action='store_true')
parser.add_argument("--syn_img_num", type=int, default=-1)
parser.add_argument("--factor", type=int, default=2)
parser.add_argument("--lamda", type=float, default=0.01)

# FedTC
parser.add_argument("--fedtc_warmup_rounds", type=int, default=20)
parser.add_argument("--fedtc_noise_init", type=float, default=0.08)
parser.add_argument("--fedtc_noise_decay", type=float, default=0.995)
parser.add_argument("--fedtc_noise_min", type=float, default=0.005)
parser.add_argument("--fedtc_tau_max", type=float, default=0.3)
parser.add_argument("--fedtc_server_update_lr", type=float, default=0.5)
parser.add_argument("--fedtc_beta", type=float, default=0.3)
parser.add_argument("--fedtc_gamma_fairness", type=float, default=0.1)
parser.add_argument("--fedtc_gamma_entropy", type=float, default=0.05)
