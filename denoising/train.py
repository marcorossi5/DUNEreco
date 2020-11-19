import sys
import os
import numpy as np

import torch
import torch.distributed as dist
from torch import optim
from torch import nn
from torch.utils.data.distributed import DistributedSampler 
from torch.utils.data import DataLoader

from model_utils import freeze_weights
from model_utils import MyDDP

from losses import get_loss

from time import time as tm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import compute_psnr

def train_epoch(args, epoch, train_loader, model, optimizer, task):
    if args.rank == 0:
        print('\n[+] Training')
    start = tm()
    loss_fn = get_loss(args.loss_fn)(args.a) if task=='dn' else \
              torch.nn.BCELoss()
    model.train()
    for i, (clear, noised) in enumerate(train_loader):
        clear = clear.to(args.dev_ids[0])
        noised = noised.to(args.dev_ids[0])
        optimizer.zero_grad()
        out = model(noised)
        idx = 0 if task=='dn' else 1 # label idx
        loss = loss_fn(out, clear[:,idx:idx+1])
        loss.backward()
        optimizer.step()

    return np.array([loss.item()]), tm() - start

def test_epoch(test_data, model, args, task, dry_inference=True):
    """
    Parameters:
        test_data: torch.utils.data.DataLoader, based on PlaneLoader
        task: str, either roi or dn
    Outputs:
        np.array: n metrics, shape (2*n)
        torch.Tensor: denoised data, shape (batch,C,W,H)
        float: dry inference time
    """
    test_sampler = DistributedSampler(dataset=test_data, shuffle=False)
    test_loader = DataLoader(dataset=test_data, sampler=test_sampler,
                              batch_size=args.test_batch_size,
                              num_workers=args.num_workers)
    if args.rank == 0:
        print('\n[+] Testing')
    model.eval()

    n = test_data.noisy.shape[0]

    start = tm()
    outs = []
    for noisy in test_loader:
        noisy = noisy.to(args.dev_ids[0])
        out =  model(noisy).data
        outs.append(out)
    outs = torch.cat(outs)
    ws = dist.get_world_size() # world size
    output = [torch.zeros_like(outs) for i in range(ws)]
    dist.all_gather(output, outs)
    h, w = args.patch_size # height, weight
    c = args.input_channels # channels
    output = torch.cat( output )
    output = output.reshape(ws,-1,c,h,w).transpose(0,1).reshape(-1,c,h,w)
    output = test_data.converter.tiles2planes( output )
    if task == 'dn':
        mask = (output <= args.threshold) & (output >= -args.threshold)
        output[mask] = 0
    end = tm()    

    if dry_inference:
        return output
    idx = 0 if task=="dn" else 1
    target = test_data.clear[:,idx:idx+1].to(args.dev_ids[0])

    def reduce(loss):
        """ Reduces losses keeping the batch size """
        return loss.reshape(n,-1).mean(-1)
    def to_np(tensor):
        """ Cast gpu torch tensor to numpy """
        return tensor.cpu().numpy()

    if args.rank==0:
        fname = "labels"
        torch.save(target.cpu(), fname)
        fname = "results"
        torch.save(output.cpu(), fname)

    loss_fn = get_loss(args.loss_fn)(args.a, reduction='none') if \
              task=='dn' else nn.BCELoss(reduction='none')
    loss = to_np(reduce( loss_fn(target, output) ))
    if task == 'dn':
        ssim = to_np(reduce( 1-get_loss('loss_ssim')(reduction='none')(target, output) ))
        mse = to_np(reduce( get_loss('mse')(reduction='none')(output, target) ))
        psnr = to_np(compute_psnr(target.cpu(), output.cpu(), reduction='none'))

        return np.array([np.mean(loss), np.std(loss)/np.sqrt(n),
                np.mean(ssim), np.std(ssim)/np.sqrt(n),
                np.mean(psnr), np.std(psnr)/np.sqrt(n),
                np.mean(mse), np.std(mse)/np.sqrt(n)]), output, end-start

    return np.array([np.mean(loss), np.std(loss)/np.sqrt(n)]), output, \
            end-start


def average_fn(x, y):
    """
    Needed to store correct metrics when training on both collection and
    induction
    """
    x = x.reshape([-1,2])
    y = y.reshape([-1,2])
    means = (x[:,0] + y[:,0])*0.5
    stds = np.sqrt((x[:,1])**2 + (y[:,1])**2)*0.5
    return np.stack([means, stds], 1).flatten()


########### main train function
def train(args, train_data, val_data, model):
    task = args.task
    # check if load existing model
    model = freeze_weights(model, task)
    model = MyDDP(model.to(args.dev_ids[0]), device_ids=args.dev_ids,
                  find_unused_parameters=True)

    if args.load:
        if args.load_path is None:
            #resume a previous training from an epoch
            #time train
            fname = os.path.join(args.dir_timings, 'timings_train.npy')
            time_train = list(np.load(fname))

            #time test
            fname = os.path.join(args.dir_timings, 'timings_test.npy')
            time_test = list(np.load(fname))

            #loss_sum
            fname = os.path.join(args.dir_metrics, 'loss_sum.npy')
            loss_sum = list(np.load(fname).T)
    
            #test_epochs
            fname = os.path.join(args.dir_metrics, 'test_epochs.npy')
            test_epochs = list(np.load(fname))

            #test metrics
            fname = os.path.join(args.dir_metrics, 'test_metrics.npy')
            test_metrics = list(np.load(fname).T)

            fname = os.path.join(args.dir_saved_models,
                                 f'{args.model}_{task}_{args.load_epoch}.dat')
            epoch = args.load_epoch + 1
       
        else:
            #load a trained model
            fname = args.load_path
            epoch = 1
            loss_sum = []
            test_metrics = []
            test_epochs = []
            time_train = []
            time_test = []

        if args.rank == 0:
            print(f'Loading model at {fname}')
        map_location = {"cuda:{0:d}": f"cuda:{args.local_rank:d}"}
        model.load_state_dict(torch.load(fname, map_location=map_location))
    else:
        epoch = 1
        loss_sum = []
        test_metrics = []
        test_epochs = []
        time_train = []
        time_test = []

    best_loss = 1e10
    best_loss_std = 0
    best_model_name = os.path.join(args.dir_saved_models,f"{args.model}_-1.dat")
        
    # initialize optimizer
    lr = args.lr_roi if (task=='roi') else args.lr_dn
    optimizer = optim.Adam(list(model.parameters()), lr=lr,
                           amsgrad=args.amsgrad)

    train_sampler = DistributedSampler(dataset=train_data, shuffle=True)
    train_loader = DataLoader(dataset=train_data, sampler=train_sampler,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers)

    # start main loop
    while epoch <= args.epochs:
        # train
        start = tm()
        train_sampler.set_epoch(epoch)
        loss, t = train_epoch(args, epoch, train_loader, model,
                          optimizer,task=task)
        end = tm() - start
        loss_sum.append(loss)
        time_train.append(t)

        if epoch % args.epoch_log == 0 and (not args.scan) and args.rank==0:
            print(f"Epoch: {epoch:3}, Loss: {loss_sum[-1][0]:6.5}, epoch time: {end:.4}s")
        # test
        if epoch % args.epoch_test == 0 and epoch>=args.epoch_test_start:
            test_epochs.append(epoch)
            start = tm()
            x, _, t = test_epoch(val_data, model, args, task,
                              dry_inference=False)
            end = tm() - start
            test_metrics.append(x)
            time_test.append(t)
            if not args.scan and args.rank==0:
                if task == 'roi':
                    print(f"Test loss on collection APAs: {x[0]:.5} +- {x[1]:.5}")
                if task == 'dn':
                    print(f"Test on collection APAs: {'loss:':7} {x[0]:.5} +- {x[1]:.5}\n \
                         {'ssim:':7} {x[2]:.5} +- {x[3]:.5}\n \
                         {'psnr:':7} {x[4]:.5} +- {x[5]:.5}\n \
                         {'mse:':7} {x[6]:.5} +- {x[7]:.5}")
                print(f'Test epoch time: {end:.4}')

            #save the model if it is the best one
            if test_metrics[-1][0] + test_metrics[-1][1] < best_loss \
               and args.rank==0:
                best_loss = test_metrics[-1][0]
                best_loss_std = test_metrics[-1][1]

                #switch to keep all the history of saved models 
                #or just the best one
                fname = os.path.join(args.dir_saved_models,
                         f'{args.model}_{task}_{epoch}.dat')
                best_model_name = fname
                bname = os.path.join(args.dir_final_test, 'best_model.txt')
                with open(bname, 'w') as f:
                    f.write(fname)
                    f.close()
                if (not args.scan) and args.rank==0:
                    print('updated best model at: ',bname)
            if args.save and args.rank==0:
                fname = os.path.join(args.dir_saved_models,
                         f'{args.model}_{task}_{epoch}.dat')
                if not args.scan:
                    print('saved model at: %s'%fname)
            if args.rank==0:
                torch.save(model.state_dict(), fname)

        epoch += 1
    if args.rank==0:
        #loss_sum
        loss_sum = np.stack(loss_sum,1)
        fname = os.path.join(args.dir_metrics, 'loss_sum')
        np.save(fname, loss_sum)
    
        #test_epochs
        fname = os.path.join(args.dir_metrics, 'test_epochs')
        np.save(fname, test_epochs)

        #test metrics
        test_metrics = np.stack(test_metrics,1)
        fname = os.path.join(args.dir_metrics, 'test_metrics')
        np.save(fname, test_metrics)

        #time train
        fname = os.path.join(args.dir_timings, 'timings_train')
        np.save(fname, time_train)

        #time test
        fname = os.path.join(args.dir_timings, 'timings_test')
        np.save(fname, time_test)

    return best_loss, best_loss_std, best_model_name
