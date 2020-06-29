import sys
import os
import torch
from torch import optim
import numpy as np
import matplotlib.pyplot as plt

from model_utils import split_img
from model_utils import recombine_img
from model_utils import plot_crops
from model_utils import plot_ROI_stats

from losses import loss_ssim

from time import time as tm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import compute_psnr

def train_epoch(args, epoch, train_data, model, optimizer):
    model.train()
    for i, (clear, noised) in enumerate(train_data):
        clear = clear.to(args.device)
        noised = noised.to(args.device)
        optimizer.zero_grad()
        loss, loss_hits, out, hits = model(noised, clear)
        loss.mean().backward()
        optimizer.step()

        out = out.cpu().detach()
        clear = clear.cpu().detach()
        hits= hits.cpu().detach()

        #plot the last crops chunk
    if args.plot_acts:
        sample = torch.randint(0,
                        out.shape[0],
                       (25,))
        plot_crops(args.dir_testing,
                out,
               "act_epoch%d_DN"%epoch,
               sample)
        plot_crops(args.dir_testing,
                hits,
               "act_epoch%d_DNhits"%epoch,
               sample)
        plot_crops(args.dir_testing,
                clear[:,:1],
               "act_epoch%d_clear"%epoch,
               sample)
        plot_crops(args.dir_testing,
                clear[:,1:2],
               "act_epoch%d_hits"%epoch,
               sample)

    return np.array([loss.mean().item()])

def test_epoch(args, epoch, test_data, model):
    """

    Outputs:
        np.array containing n metrics, shape (2*n)
        torch.tensor containing denoised data, shape (batch,C,W,H)
    """
    model.eval()
    mse = []
    psnr = []
    loss = []
    res = []
    ssim = []

    for clear, noisy, norm in test_data:
        hits = clear[:,1:2].to(args.device)
        clear = clear[:,:1].to(args.device)        
        noisy = noisy.to(args.device)
        norm = norm[0].to(args.device)
        crops, crops_shape, pad = split_img(noisy,model.patch_size)
        loader = torch.split(crops,args.test_batch_size)
        dn = []
        for chunk in loader:
            answer = model(chunk.to(args.device)).data
            dn.append(answer)
        dn = torch.cat(dn)
        dn = recombine_img(dn, crops_shape, pad)
        dn_hits = dn[:,1:2]
        dn = dn[:,:1] * (norm[1]-norm[0]) + norm[0]
        if args.model == 'ROI':
            loss.append(model.xent(hits,dn_hits).cpu().item())
        else:
            loss.append((model.loss_fn(clear,dn)
                    +model.xent(hits,dn_hits)).cpu().item())
        ssim.append(1-loss_ssim()(clear,dn).cpu().item())
        mse.append(torch.nn.MSELoss()(clear,dn).cpu().item())
        psnr.append(compute_psnr(clear,dn))
        res.append(dn.cpu().detach())
    res = torch.cat(res)

    if args.model == 'ROI':
        plot_ROI_stats(args,epoch,hits,dn_hits,args.t)

    n = len(loss)
    return np.array([np.mean(loss), np.std(loss)/np.sqrt(n),
                np.mean(ssim), np.std(ssim)/np.sqrt(n),
                np.mean(psnr), np.std(psnr)/np.sqrt(n),
                np.mean(mse), np.std(mse)/np.sqrt(n)]), res

########### main train function
def train(args, train_data, test_data, model):
    # check if load existing model
    if args.load:
        fname = os.path.join(args.dir_saved_models,
            args.model + '_%d'%args.load_epoch + '.dat')
        model.load_state_dict(torch.load(fname))

        args.lr = 5e-4
        epoch = args.load_epoch

        fname = os.path.join(args.dir_timings, 'all_timings.npy')
        time_all = list(np.load(fname))

        #loss_sum
        fname = os.path.join(args.dir_metrics, 'loss_sum.npy')
        loss_sum = list(np.load(fname).T)
    
        #test_epochs
        fname = os.path.join(args.dir_metrics, 'test_epochs.npy')
        test_epochs = list(np.load(fname))

        #test metrics
        fname = os.path.join(args.dir_metrics, 'test_metrics.npy')
        test_metrics = list(np.load(fname).T)
        print('model loaded!, lr: {}'.format(args.lr))
    else:
        epoch = 1
        loss_sum = []
        test_metrics = []
        test_epochs = []
    best_loss = 1e10
    best_loss_std = 0
    best_model_name = os.path.join(args.dir_saved_models,
                             f'{args.model}_-1.dat')
        
    # initialize optimizer
    optimizer=  optim.Adam(list(model.parameters()), lr=args.lr,
                           amsgrad=args.amsgrad)
    
    # start main loop
    while epoch <= args.epochs:
        time_start = tm()
        # train
        loss = train_epoch(args, epoch, train_data, model,
                          optimizer)
        loss_sum.append(loss)

        time_end = tm()-time_start
        if epoch % args.epoch_log == 0 and (not args.scan):
            print("\nEpoch: %d, Loss: %.5f, time: %.5f"%(epoch,
                                                      loss_sum[-1][0],
                                                      time_end))
        # test
        if epoch % args.epoch_test == 0 and epoch>=args.epoch_test_start:
            print('test start ...')
            test_epochs.append(epoch)
            start = tm()
            x, _ = test_epoch(args, epoch, test_data, model)
            test_metrics.append(x)
            if not args.scan:
                print('Test loss: %.5f +- %.5f,\
                       ssim: %.5f +- %.5f,\
                       psnr: %.5f +- %.5f,\
                       mse: %.5e +- %.5e'%(x[0], x[1], x[2], x[3],
                                           x[4], x[5], x[6], x[7]))
                print('Test time: %.4f\n'%(tm()-start))
            

            #save the model if it is the best one
            if test_metrics[-1][0] + test_metrics[-1][1] < best_loss:
                best_loss = test_metrics[-1][0]
                best_loss_std = test_metrics[-1][1]

                #switch to keep all the history of saved models 
                #or just the best one
                if args.save:
                    fname = os.path.join(args.dir_saved_models,
                             f'{args.model}_{epoch}.dat')
                else:
                    fname = os.path.join(args.dir_saved_models,
                             f'{args.model}.dat')
                torch.save(model.state_dict(), fname)
                if not args.scan:
                    print('saved model at: %s'%fname)
                best_model_name = fname
                bname = os.path.join(args.dir_final_test, 'best_model.txt')
                with open(bname, 'w') as f:
                    f.write(fname)
                    f.close()
                if not args.scan:
                    print('updated best model at: ',bname)

        epoch += 1
    
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

    return best_loss, best_loss_std, best_model_name
