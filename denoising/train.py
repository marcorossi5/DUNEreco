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

def train_epoch(args, epoch, train_loader, model, optimizer, warmup):
    if args.rank == 0:
        print('\n[+] Training')
    start = tm()
    loss_fn = get_loss(args.loss_fn)(args.a) if warmup=='dn' else \
              torch.nn.BCELoss()
    model.train()
    for i, (clear, noised) in enumerate(train_loader):
        clear = clear.to(args.dev_ids[0])
        noised = noised.to(args.dev_ids[0])
        optimizer.zero_grad()
        out = model(noised)
        idx = 0 if warmup=='dn' else 1 # label idx
        loss = loss_fn(out, clear[:,idx:idx+1])
        loss.backward()
        optimizer.step()

    return np.array([loss.item()]), tm() - start

def test_epoch(test_data, model, args, warmup, dry_inference=True):
    """
    Parameters:
        test_data: torch.utils.data.DataLoader, based on PlaneLoader
        warmup: str, either roi or dn
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
        print('[+] Testing')
    model.eval()

    n = test_data.noisy.shape[0]

    start = tm()
    outs = []
    for noisy in test_loader:
        noisy = noisy.to(args.dev_ids[0])
        out =  model(noisy).data
        outs.append(out)
    outs = torch.cat(outs)
    output = [torch.zeros_like(outs) for i in range(dist.get_world_size())]
    dist.all_gather(output, outs)
    output = test_data.converter.tiles2planes( torch.cat(output) )
    if warmup == 'dn':
        output = test_data.converter.invert_normalization(output)
        output [output <= args.threshold] = 0
    end = tm()    

    if dry_inference:
        return output
    idx = 0 if warmup=="dn" else 1
    target = test_data.clear[:,idx:idx+1].to(args.dev_ids[0])

    def reduce(loss):
        """ Reduces losses keeping the batch size """
        return loss.reshape(n,-1).mean(-1)
    def to_np(tensor):
        """ Cast gpu torch tensor to numpy """
        return tensor.cpu().numpy()

    loss_fn = get_loss(args.loss_fn)(args.a, size_average=False) if \
              warmup=='dn' else nn.BCELoss(reduction='none')
    loss = to_np(reduce( loss_fn(target, output) ))
    if warmup == 'dn':
        ssim = to_np(reduce( 1-loss_ssim(size_average=False)(target, output) ))
        mse = to_np(reduce( nn.MSELoss(reduction='none')(output, target) ))
        psnr = to_np(compute_psnr(target, output, reduction='none'))

        return np.array([np.mean(loss), np.std(loss)/np.sqrt(n),
                np.mean(ssim), np.std(ssim)/np.sqrt(n),
                np.mean(psnr), np.std(psnr)/np.sqrt(n),
                np.mean(mse), np.std(mse)/np.sqrt(n)]), output, end-start

    return np.array([np.mean(loss), np.std(loss)/np.sqrt(n)]), output, \
            end-start


########### main train function
def train(args, train_data, val_data, model):
    warmup = args.warmup
    # check if load existing model
    model = freeze_weights(model, warmup)
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
                                 f'{args.model}_{warmup}_{args.load_epoch}.dat')
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
        map_location = {'cuda:%d' % 0: 'cuda:%d' % args.rank}
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
    best_model_name = os.path.join(args.dir_saved_models,
                             f'{args.model}_-1.dat')
        
    # initialize optimizer
    lr = args.lr_roi if (warmup=='roi') else args.lr_dn
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
                          optimizer,warmup=warmup)
        end = tm() - start
        loss_sum.append(loss)
        time_train.append(t)

        if epoch % args.epoch_log == 0 and (not args.scan) and args.rank==0:
            print("Epoch: %d, Loss: %.5f, epoch time: %.5f"%(epoch,
                                                      loss_sum[-1][0],
                                                      end))
        # test
        if epoch % args.epoch_test == 0 and epoch>=args.epoch_test_start:
            if args.rank == 0:
                print('test start ...')
            test_epochs.append(epoch)
            start = tm()
            x, _, t = test_epoch(val_data, model, args, warmup,
                              dry_inference=False)
            end = tm() - start
            test_metrics.append(x)
            time_test.append(t)
            if not args.scan and args.rank==0:
                if warmup == 'roi':
                    print('Test loss: %.5f +- %.5f'%(x[0], x[1]))
                if warmup == 'dn':
                    print('Test loss: %.5f +- %.5f,\
                           ssim: %.5f +- %.5f,\
                           psnr: %.5f +- %.5f,\
                           mse: %.5e +- %.5e'%(x[0], x[1], x[2], x[3],
                                           x[4], x[5], x[6], x[7]))
                print('Test epoch time: %.4f\n'% end)            

            #save the model if it is the best one
            if test_metrics[-1][0] + test_metrics[-1][1] < best_loss \
               and args.rank==0:
                best_loss = test_metrics[-1][0]
                best_loss_std = test_metrics[-1][1]

                #switch to keep all the history of saved models 
                #or just the best one
                fname = os.path.join(args.dir_saved_models,
                         f'{args.model}_{warmup}_{epoch}.dat')
                best_model_name = fname
                bname = os.path.join(args.dir_final_test, 'best_model.txt')
                with open(bname, 'w') as f:
                    f.write(fname)
                    f.close()
                if (not args.scan) and args.rank==0:
                    print('updated best model at: ',bname)
            if args.save and args.rank==0:
                fname = os.path.join(args.dir_saved_models,
                         f'{args.model}_{warmup}_{epoch}.dat')
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
