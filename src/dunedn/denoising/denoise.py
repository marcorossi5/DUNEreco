# This file is part of DUNEdn by M. Rossi
import os
import argparse
import time as tm

from dunedn.denoising.distributed import set_random_seed
from dunedn.denoising.dataloader import PlaneLoader, CropLoader
from dunedn.denoising.model import SCG_Net, DenoisingModel
from dunedn.denoising.args import Args
from dunedn.denoising.model_utils import print_summary_file
from dunedn.denoising.train import train
from dunedn.utils.utils import load_yaml


def main(args):
    # n = torch.cuda.device_count() // args.local_world_size
    # args.dev_ids = list(range(args.local_rank * n, (args.local_rank + 1) * n))
    # args.dev_ids = [args.dev]
    args.dev_ids = [0, 1, 2, 3]

    if args.model == "uscg":
        model = SCG_Net(task=args.task, h=args.patch_h, w=args.patch_w)
    elif args.model in ["cnn", "gcnn"]:
        args.patch_size = eval(args.patch_size)
        model = DenoisingModel(
            args.model,
            args.task,
            args.channel,
            args.patch_size,
            args.input_channels,
            args.hidden_channels,
            args.k,
            args.dataset_dir,
            args.normalization,
        )
    else:
        raise NotImplementedError("Model not implemented")

    # load datasets
    set_random_seed(0)
    loader = PlaneLoader if args.model == "uscg" else CropLoader
    kwargs = {} if args.model == "uscg" else {"patch_size": args.patch_size}
    train_data = loader(
        args.dataset_dir, "train", args.task, args.channel, args.threshold
    )
    val_data = PlaneLoader(
        args.dataset_dir, "val", args.task, args.channel, args.threshold, **kwargs
    )

    # train
    return train(args, train_data, val_data, model)


def spmd_main(card, local_rank, local_world_size, dev):
    """ Spawn distributed processes """
    # dist.init_process_group(backend="nccl")

    prefix = "/nfs/public/romarco/DUNEreco/denoising/configcards"
    parameters = load_yaml(os.path.join(prefix, card))
    parameters["local_rank"] = local_rank
    parameters["local_world_size"] = local_world_size
    parameters["rank"] = 0
    parameters["dev"] = dev
    # parameters["rank"] = dist.get_rank()
    args = Args(**parameters)
    args.build_directories(build=(args.rank == 0))
    if args.rank == 0:
        print_summary_file(args)
    START = tm.time()
    main(args)
    if args.rank == 0:
        print(f"[{os.getpid()}] Process done in {tm.time()-START}")

    # dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--card", type=str, help="yaml config file path", default="default_config.yaml"
    )
    parser.add_argument("--local_rank", default=0, type=int, help="Distributed utility")
    parser.add_argument(
        "--local_world_size", default=1, type=int, help="Distributed utility"
    )
    parser.add_argument("--dev", default=0, type=int, help="cuda device")
    # load configuration
    args = vars(parser.parse_args())
    # main
    spmd_main(**args)
