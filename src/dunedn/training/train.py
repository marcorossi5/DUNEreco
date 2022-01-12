# This file is part of DUNEdn by M. Rossi
"""
    This module implements the main training loops and inference functions.
"""
import logging
from math import ceil, sqrt
from time import time as tm
import numpy as np

import torch
from torch import optim
from torch.utils.data import DataLoader

from dunedn.configdn import PACKAGE
from dunedn.training.losses import get_loss
from dunedn.networks.model_utils import model2batch

# instantiate logger
logger = logging.getLogger(PACKAGE + ".train")


def time_windows(planes, w, stride):
    """
    Takes time windows of given width and stride from a planes.

    Parameters
    ----------
        - planes: torch.Tensor, planes of shape=(N,C,H,W)
        - w: int, width of the time windows
        - stride: int, steps between time windows

    Returns
    -------
        - list, masks for time windows each of shape [N,C,H,w]
        - list, time windows each of shape [N,C,H,w]
        - list, each elements is [start idx, end idx] of the time window
    """
    _, _, _, W = planes.size()
    n = ceil((W - w) / stride) + 1
    base = np.arange(n).reshape([-1, 1]) * stride
    idxs = [[0, w]] + base
    windows = []
    div = torch.zeros_like(planes)
    for start, end in idxs:
        windows.append(planes[:, :, :, start:end])
        div[:, :, :, start:end] += 1
    return div, windows, idxs


def train_uscg_epoch(args, epoch, train_loader, model, optimizer, balance_ratio, task):
    """
    Train USCG network.

    Parameters
    ----------
        - args: Args, runtime settings
        - epoch: int, epoch number
        - train_loader: torch.utils.data.DataLoader, train data loader
        - model: MyDataParallel, the model to train
        - optimizer: torch.optim.Adam, the Adam optimizer
        - balance_ratio: float, categories balancing (roi only, None for dn).
        - task: str, available options dn | roi

    Returns
    -------
        - np.array, training epoch loss of shape=(1,)
        - float, training epoch time
    """
    if args.rank == 0:
        logger.debug("Training epoch")
    start = tm()
    loss_fn = (
        get_loss(args.loss_fn)(args.a)
        if task == "dn"
        else get_loss("bce")(balance_ratio)
    )
    model.train()
    for noisy, clear in train_loader:
        _, cwindows, _ = time_windows(clear, args.patch_w, args.patch_stride)
        _, nwindows, _ = time_windows(noisy, args.patch_w, args.patch_stride)
        for nwindow, cwindow in zip(nwindows, cwindows):
            cwindow = cwindow.to(args.dev[0])
            nwindow = nwindow.to(args.dev[0])
            optimizer.zero_grad()
            out, loss0 = model(nwindow)
            loss1 = loss_fn(out, cwindow)
            loss = loss1 + loss0
            loss.backward()
            optimizer.step()
    return np.array([loss.item()]), tm() - start


def train_gcnn_epoch(args, epoch, train_loader, model, optimizer, balance_ratio, task):
    """
    Train USCG network.

    Parameters
    ----------
        - args: Args, runtime settings
        - epoch: int, epoch number
        - train_loader: torch.utils.data.DataLoader, train data loader
        - model: MyDataParallel, the model to train
        - optimizer: torch.optim.Adam, the Adam optimizer
        - balance_ratio: float, categories balancing (roi only, None for dn).
        - task: str, available options dn | roi

    Returns
    -------
        - np.array, training epoch loss of shape=(1,)
        - float, training epoch time
    """
    logger.debug("Training epoch")
    start = tm()
    model.train()
    loss_fn = (
        get_loss(args.loss_fn)(args.a)
        if task == "dn"
        else get_loss("bce")(balance_ratio)
    )
    for clear, noisy in train_loader:
        idx = 0 if task == "dn" else 1
        clear = clear[:, idx : idx + 1].to(args.dev[0])
        noisy = noisy.to(args.dev[0])
        optimizer.zero_grad()
        out = model(noisy)
        loss = loss_fn(out, clear)
        loss.mean().backward()
        optimizer.step()
    return np.array([loss.mean().item()]), tm() - start


def choose_train(modeltype, *args):
    """
    Utility function to retrieve training from model name and args.

    Parameters
    ----------
        - modeltype: str, available options cnn | gcnn | uscg
        - args: list, model's __init__ arguments

    Returns
    -------
        - func, training function

    Raises
    ------
        - NotImplementedError if modeltype is not in ['uscg', 'cnn', 'gcnn']
    """
    if modeltype == "uscg":
        return train_uscg_epoch(*args)
    elif modeltype in ["gcnn", "cnn"]:
        return train_gcnn_epoch(*args)
    raise NotImplementedError("Model not implemented")


def inference(test_loader, stride, model, dev):
    """
    Parameters
    ----------
        - test_loader: torch.utils.data.DataLoader, inference loader
        - stride: int, steps between time windows
        - model: MyDataParallel, the model for inference
        - dev: str, the host device

    Returns
    -------
        - torch.Tensor, output tensor of shape=(N,C,H,W)
    """
    w = model.w
    model.eval()
    outs = []
    for noisy, _ in test_loader:
        div, nwindows, idxs = time_windows(noisy, w, stride)
        out = torch.zeros_like(noisy)
        for nwindow, (start, end) in zip(nwindows, idxs):
            nwindow = nwindow.to(dev)
            out[:, :, :, start:end] += model(nwindow).data
        outs.append(out / div)
    return torch.cat(outs)


def identity_inference(test_loader):
    """
    Parameters
    ----------
        - test_loader: torch.utils.data.DataLoader, dummy inference loader

    Returns
    -------
        - torch.Tensor, input noisy planes of shape=(N,C,H,W)
    """
    outs = [noisy for noisy, _ in test_loader]
    return torch.cat(outs)


def gcnn_inference(test_loader, model, dev):
    """
    Parameters
    ----------
        - test_loader: torch.utils.data.DataLoader, inference loader
        - model: MyDataParallel, the model for inference
        - dev: str, the host device

    Returns
    -------
        - torch.Tensor, output tensor of shape=(N,C,H,W)
    """
    outs = []
    for noisy, _ in test_loader:
        noisy = noisy.to(dev)
        out = model(noisy).data
        outs.append(out)
    return torch.cat(outs)


def compute_val_loss(test_loader, outputs, args, task):
    """
    Computes validation losses.

    Parameters
    ----------
        - test_loader: torch.utils.data.DataLoader, validation loader
        - outputs: torch.Tensor, output tensor of shape=(N,C,H,W)
        - args: Args, runtime settings
        - task: str, available options dn | roi

    Returns
    -------
        - list, of computed metrics with their uncertainties
                [loss, loss unc, ssim, ssim unc, psnr, psnr unc, mse, mse unc]
    """
    # if task == 'roi':
    #     metrics = ['bce', 'softdice']
    # elif task == 'dn':
    #     metrics = [args.loss_fn, 'ssim', 'psnr', 'mse']
    # else:
    #     raise NotImplementedError("Task not implemented")
    # metrics_fns = list(map(lambda x: get_loss(x)(reduction='none'), metrics))
    loss = []
    ssim = []
    mse = []
    psnr = []
    loss_fn = get_loss(args.loss_fn)(args.a) if task == "dn" else get_loss("bce")(0.5)
    if task == "dn":
        ssim_fn = get_loss("ssim")()
        mse_fn = get_loss("mse")()
        psnr_fn = get_loss("psnr")()
    for (_, target), output in zip(test_loader, outputs):
        # works only with unit batch_size
        output = output.unsqueeze(0).to(args.dev[0])
        if args.model in ["gcnn", "cnn"]:
            idx = 0 if task == "dn" else 1
            target = target[:, idx : idx + 1]
        target = target.to(args.dev[0])
        loss.append(loss_fn(output, target).unsqueeze(0))
        if task == "dn":
            ssim.append(1 - ssim_fn(output, target).unsqueeze(0))
            mse.append(mse_fn(output, target).unsqueeze(0))
            psnr.append(psnr_fn(output, target).unsqueeze(0))
    if args.rank == 0:
        fname = args.dir_testing / "labels"
        torch.save(target.cpu(), fname)
        fname = args.dir_testing / "results"
        torch.save(output.cpu(), fname)

    def all_gather(loss):
        # loss = torch.cat(loss)
        # ws = dist.get_world_size()
        # lossws = [torch.zeros_like(loss) for i in range(ws)]
        # dist.all_gather(lossws, loss)
        # return torch.cat(lossws).cpu().numpy()
        return torch.cat(loss).cpu().numpy()

    loss = all_gather(loss)

    sqrtn = sqrt(len(loss))
    if task == "dn":
        ssim = all_gather(ssim)
        psnr = all_gather(psnr)
        mse = all_gather(mse)
        return np.array(
            [
                loss.mean(),
                loss.std() / sqrtn,
                ssim.mean(),
                ssim.std() / sqrtn,
                psnr.mean(),
                psnr.std() / sqrtn,
                mse.mean(),
                mse.std() / sqrtn,
            ]
        )
    return np.array([loss.mean(), loss.std() / sqrtn])


def test_epoch(test_data, model, args, task, dry_inference=True):
    """
    Consume test data and output metrics.

    Parameters
    ----------
        - test_data: torch.utils.data.DataLoader, based on PlaneLoader
        - model: MyDataParallel, the model for inference
        - args: Args, runtime settings
        - task: str, available options dn | roi
        - dry_inference: bool, if true do not compute metrics

    Returns
    -------
        - list, of computed metrics with their uncertainties
        - torch.Tensor, output tensor of shape=(N,C,H,W)
        - float, dry inference time
    """
    if args.model in ["cnn", "gcnn"]:
        test_data.to_crops()
    # test_sampler = DistributedSampler(dataset=test_data, shuffle=False)
    test_loader = DataLoader(
        dataset=test_data,  # sampler=test_sampler,
        batch_size=model2batch[args.model][args.task],
    )
    if args.rank == 0:
        logger.debug("Testing epoch")
    start = tm()
    if args.model == "uscg":
        outputs = inference(test_loader, args.patch_stride, model, args.dev[0])
    elif args.model in ["cnn", "gcnn"]:
        outputs = gcnn_inference(test_loader, model, args.dev[0])
        outputs = test_data.converter.tiles2planes(outputs)
    # if task == 'dn':
    #     mask = (outputs <= args.threshold) & (outputs >= -args.threshold)
    #     outputs[mask] = 0
    dry_time = tm() - start

    if dry_inference:
        return outputs, dry_time
    if args.model in ["cnn", "gcnn"]:
        test_data.to_planes()
        test_loader = DataLoader(
            dataset=test_data,  # sampler=test_sampler,
            batch_size=1,
        )
    return compute_val_loss(test_loader, outputs, args, task), outputs, dry_time


def optimizer_fn(params, lr, amsgrad):
    """
    Utility function to retrieve optimizer.

    Parameters
    ----------
        - params: list, parameters to optimize
        - lr: float, learning rate
        - amsgrad: bool, whether to use the AMSGrad variant of this algorithm

    Returns
    -------
        - torch.optim.Adam, the optimizer
    """
    return optim.Adam(params, lr=lr, amsgrad=amsgrad)


########### main train function
def train(args, train_data, val_data, model):
    """
    Training function.

    Parameters
    ----------
        - args: Args, runtime settings
        - train_data: CropLoader | PlaneLoader, training data loader
        - val_data: PlaneLoader, validation data loader
        - model: MyDataParallel, the model to train

    Returns
    -------
        - float, minimum loss over training
        - float, uncertainty over minimum loss
        - str, best checkpoint file name
    """
    task = args.task
    channel = args.channel
    # check if load existing model
    # model = MyDataParallel(model, device_ids=args.dev)
    model = model.to(args.dev[0])
    # model = MyDDP(model.to(args.dev[0]), device_ids=args.dev,
    #               find_unused_parameters=True)

    if args.load:
        if args.load_path is None:
            # resume a previous training from an epoch
            # time train
            fname = args.dir_timings / "timings_train.npy"
            time_train = list(np.load(fname))

            # time test
            fname = args.dir_timings / "timings_test.npy"
            time_test = list(np.load(fname))

            # loss_sum
            fname = args.dir_metrics / "loss_sum.npy"
            loss_sum = list(np.load(fname).T)

            # test_epochs
            fname = args.dir_metrics / "test_epochs.npy"
            test_epochs = list(np.load(fname))

            # test metrics
            fname = args.dir_metrics / "test_metrics.npy"
            test_metrics = list(np.load(fname).T)

            fname = args.dir_saved_models / f"{args.model}_{task}_{args.load_epoch}.dat"
            epoch = args.load_epoch + 1

        else:
            # load a trained model
            fname = args.load_path
            epoch = 1
            loss_sum = []
            test_metrics = []
            test_epochs = []
            time_train = []
            time_test = []

        if args.rank == 0:
            logger.info("Loading model at %s", fname)
        # map_location = {"cuda:{0:d}": f"cuda:{args.dev:d}"}
        # map_location = {"cuda:{0:d}": f"cuda:{args.local_rank:d}"}
        # model.load_state_dict(torch.load(fname, map_location=map_location))
        model.load_state_dict(torch.load(fname))
    else:
        epoch = 1
        loss_sum = []
        test_metrics = []
        test_epochs = []
        time_train = []
        time_test = []

    best_loss = 1e10
    best_loss_std = 0
    best_model_name = args.dir_saved_models / f"{args.model}_-1.dat"

    # initialize optimizer
    # lr = args.lr_roi if task == "roi" else args.lr_dn
    lr = args.lr
    optimizer = optimizer_fn(list(model.parameters()), lr, args.amsgrad)

    # train_sampler = DistributedSampler(dataset=train_data, shuffle=True)
    train_loader = DataLoader(
        dataset=train_data,
        shuffle=True,  # sampler=train_sampler,
        batch_size=args.batch_size,
    )

    # main training loop
    while epoch <= args.epochs:
        if epoch % 6 == 0:
            lr = args.lr_dn if task == "dn" else args.lr_roi
            optimizer = optimizer_fn(list(model.parameters()), lr, args.amsgrad)

        # train
        start = tm()
        # train_sampler.set_epoch(epoch)
        balance_ratio = train_data.balance_ratio if task == "roi" else None
        loss, t = choose_train(
            args.model, args, epoch, train_loader, model, optimizer, balance_ratio, task
        )
        end = tm() - start
        loss_sum.append(loss)
        time_train.append(t)
        if epoch % args.epoch_log == 0 and (not args.scan) and args.rank == 0:
            logger.info(
                "Epoch: %3d, Loss: %6.5f, epoch time: %.4fs",
                epoch,
                loss_sum[-1][0],
                end,
            )

        # test
        if epoch % args.epoch_test == 0 and epoch >= args.epoch_test_start:
            test_epochs.append(epoch)
            start = tm()
            x, _, t = test_epoch(val_data, model, args, task, dry_inference=False)
            end = tm() - start
            test_metrics.append(x)
            time_test.append(t)
            if args.rank == 0:
                if args.task == "roi":
                    logger.info(
                        "Test loss on %10s APAs: %.5f +- %.5f", channel, *x[0:2]
                    )
                if args.task == "dn":
                    logger.info(
                        f"Test on {channel:10} APAs: {'loss:':7} {x[0]:.5} +- {x[1]:.5}\n\
                         {'ssim:':7} {x[2]:.5} +- {x[3]:.5}\n\
                         {'psnr:':7} {x[4]:.5} +- {x[5]:.5}\n\
                         {'mse:':7} {x[6]:.5} +- {x[7]:.5}"
                    )
                logger.info("Test epoch time: %.4f", end)

            # save the model if it is the best one
            if test_metrics[-1][0] + test_metrics[-1][1] < best_loss and args.rank == 0:
                best_loss = test_metrics[-1][0]
                best_loss_std = test_metrics[-1][1]

                # switch to keep all the history of saved models
                # or just the best one
                fname = args.dir_saved_models / f"{args.model}_{task}_{epoch}.dat"
                best_model_name = fname
                bname = args.dir_final_test / "best_model.txt"
                with open(bname, "w") as f:
                    f.write(fname.as_posix())
                    f.close()
                if (not args.scan) and args.rank == 0:
                    logger.info("Updated best model at: %s", bname)
            if args.save and args.rank == 0:
                fname = (
                    args.dir_saved_models / f"{args.model}_{task}_{channel}_{epoch}.dat"
                )
                if not args.scan:
                    logger.info("Saved model at: %s", fname)
            if args.rank == 0:
                torch.save(model.state_dict(), fname)

        epoch += 1
    if args.rank == 0:
        # loss_sum
        loss_sum = np.stack(loss_sum, 1)
        fname = args.dir_metrics / "loss_sum"
        np.save(fname, loss_sum)

        # test_epochs
        fname = args.dir_metrics / "test_epochs"
        np.save(fname, test_epochs)

        # test metrics
        test_metrics = np.stack(test_metrics, 1)
        fname = args.dir_metrics / "test_metrics"
        np.save(fname, test_metrics)

        # time train
        fname = args.dir_timings / "timings_train"
        np.save(fname, time_train)

        # time test
        fname = args.dir_timings / "timings_test"
        np.save(fname, time_test)

    return best_loss, best_loss_std, best_model_name


# TODO: fix the validation loss, it must be the same of the training loss
