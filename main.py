import os
import swanlab as wandb
import random
from config import parser
from src.servers.serveravg import FedAvg
from src.servers.serveravgm import FedAvgM
from src.servers.serverdm import FedDM
from src.servers.serverdyn import FedDyn
from src.servers.servergen import FedGen
from src.servers.serverlc import FedLC
from src.servers.servermoon import MOON
from src.servers.serverpad import FedPAD
from src.servers.serverprox import FedProx
from src.servers.serverscaffold import SCAFFOLD
from src.utils.train_utils import setup_seed, get_network, get_logger
from datasets.utils.dataset_utils import get_dataset_info


def main():
    args = parser.parse_args()
    split_file = f'/{args.dataset}_num_clients={args.num_clients}_alpha={args.dir_alpha}.json'
    args.split_file = os.path.join(os.path.dirname(__file__), "datasets/split_file" + split_file)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    args.seed = random.randint(1, 10000)
    setup_seed(args.seed)

    model_identification = f'{args.algorithm}/{args.dataset}_dir{args.dir_alpha}_{args.num_clients}clients/'

    if args.algorithm in ['FedDM', 'FedAF', 'FedPAD']:
        wandb_name = f'{args.model}_{args.dc_iterations}dc_{args.global_model_epochs}epochs_{args.image_lr}imglr'
        wandb_name += f'_balance_{args.ipc}ipc' if args.balance else f'_unbalance_{args.ipc}ipc'
        if args.algorithm == 'FedPAD':
            wandb_name += f"_{args.lamda}lamda"

        if args.init == 'noise':
            wandb_name += f'_noise_{args.sigma}sigma_{args.gnb}gnb'
        elif args.init == 'real':
            wandb_name += f'_real'
        else:
            wandb_name += f'_mix_{args.factor}factor'
    else:
        wandb_name = f'{args.model}_{args.local_epochs}epochs_{args.local_batch_size}lbs_{args.local_learning_rate}lr'


    model_identification += "debug" if args.mode == "debug" else wandb_name
    args.save_folder_name = os.path.join('results', model_identification)
    args.model_str = args.model

    mode = "disabled" if args.mode != "online" else "online"
    wandb.init(
        project=f'FedPAD',
        name=f"{args.algorithm}_{args.dataset}_client{args.num_clients}_dir{args.dir_alpha}_{wandb_name}",
        mode=mode,
    )
    wandb.config.update(args)

    # swanlab not supports
    p_flag = True if args.mode in ["debug", "local"] else False
    args.logger = get_logger(args.save_folder_name, p_flag)
    args.logger.info(f"Save folder: {args.save_folder_name}")
    args.dataset_info = get_dataset_info(args.dataset, "datasets/data/")
    args.model = get_network(args.model, args.dataset_info).to(args.device)

    if args.algorithm == 'FedAvg':
        server = FedAvg(args, 1)
    elif args.algorithm == 'FedAvgM':
        server = FedAvgM(args, 1)
    elif args.algorithm == 'FedProx':
        server = FedProx(args, 1)
    elif args.algorithm == 'SCAFFOLD':
        server = SCAFFOLD(args, 1)
    elif args.algorithm == 'MOON':
        server = MOON(args, 1)
    elif args.algorithm == 'FedDyn':
        server = FedDyn(args, 1)
    elif args.algorithm == 'FedGen':
        server = FedGen(args, 1)
    elif args.algorithm == 'FedLC':
        server = FedLC(args, 1)
    elif args.algorithm == 'FedDM':
        server = FedDM(args, 1)
    elif args.algorithm == 'FedAF':
        raise NotImplementedError()
    elif args.algorithm == 'FedPAD':
        server = FedPAD(args, 1)
    else:
        raise NotImplementedError(f"Algorithm {args.algorithm} not implemented")

    for arg in sorted(vars(args)):
        args.logger.info(f"{arg}: {getattr(args, arg)}")

    # local_params = ["local_batch_size", "local_learning_rate", "local_momentum", "local_weight_decay"]
    # global_params = ["global_batch_size", "global_learning_rate", "global_momentum", "global_weight_decay"]
    # for local_param, global_param in zip(local_params, global_params):
    #     print(getattr(args, local_param), getattr(args, global_param))
    #     assert getattr(args, local_param) == getattr(args, global_param)
    server.fit()


if __name__ == "__main__":
    main()
