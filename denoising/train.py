import sys
import os
import torch
from torch import optim
import numpy as np
import matplotlib.pyplot as plt

from model_utils import split_img
from model_utils import recombine_img
from model_utils import plot_crops
import ssim

import time as tm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import compute_psnr

def train_epoch(args, epoch, train_data, model, optimizer, scheduler, mse_loss):
    model.train()
    for i, (clear, noised) in enumerate(train_data):
        clear = clear.to(args.device)
        noised = noised.to(args.device)
        optimizer.zero_grad()
        denoised_img, loss = model(noised, clear)
        loss.mean().backward()
        optimizer.step()
    if epoch > args.warmup_epoch:
        scheduler.step()
    return np.array([loss.mean().item()])


def test_epoch(args, epoch, test_data, model, mse_loss):
    model.eval()
    mse = []
    psnr = []
    ssim_loss = []

    mse_loss = torch.nn.MSELoss(reduction='none')

    for i, (clear, noised) in enumerate(test_data):
        clear = clear.to(args.device)
        noised = noised.to(args.device)

        denoised_img = model(noised)

        ssim_loss.append((1 - ssim.ssim(denoised_img,
                                   clear,
                                   data_range=1.,
                                   size_average=True)).cpu().item())
        
        mse_ = mse_loss(denoised_img, clear).mean([-1,-2])[:,0]

        mse.append(mse_.mean().cpu().item())
        m = clear.max(-1).values.max(-1).values

        res = (m/mse_).mean().cpu().detach().numpy()
        psnr.append(10*np.log10(res))

    sample = torch.randint(0,
                           denoised_img.shape[0],
                           (25,)).cpu().detach().numpy()
    plot_crops(args.dir_testing,
               denoised_img.cpu().detach().numpy()[:,0],
               "act_epoch%d_DN"%epoch,
               sample)
    plot_crops(args.dir_testing,
               clear.cpu().detach().numpy()[:,0],
               "act_epoch%d_label"%epoch,
               sample)


    return np.array([np.array(ssim_loss), np.mean(psnr), np.std(psnr)/np.sqrt(i+1),
                np.mean(mse), np.std(mse)/np.sqrt(i+1)])

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
    best_ssim = 1
        
    # initialize optimizer
    optimizer=  optim.Adam(list(model.parameters()), lr=args.lr)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer,
                                            lambda x: args.decay_lr**x)

    # start main loop
    time_all = np.zeros(args.epochs)
    mse_loss = torch.nn.MSELoss()
    
    while epoch <= args.epochs:
        time_start = tm.time()
        # train
        loss = train_epoch(args, epoch, train_data, model,
                          optimizer, scheduler, mse_loss)
        loss_sum.append(loss)

        time_end = tm.time()
        time_all[epoch - 1] = time_end - time_start
        if epoch % args.epoch_log == 0:
            print("\nEpoch: %d, Loss: %.5f, time: %.5f"%(epoch,
                                                      loss_sum[-1][0],
                                                      time_all[epoch - 1]))
        # test
        if epoch % args.epoch_test == 0 and epoch>=args.epoch_test_start:
            print('test start ...')
            test_epochs.append(epoch)
            start = tm.time()
            test_metrics.append(test_epoch(args, epoch, test_data,
                                           model, mse_loss))
            
            print('Test ssim: %.5f,\
                   psnr: %.5f +- %.5f,\
                   mse: %.5e +- %.5e'%(test_metrics[-1][0],
                                       test_metrics[-1][1],
                                       test_metrics[-1][2],
                                       test_metrics[-1][3],
                                       test_metrics[-1][4]))
            print('Test time: %.4f\n'%(tm.time()-start))

        # save model checkpoint
        if args.save:
            if epoch % args.epoch_save == 0:
                fname = os.path.join(args.dir_saved_models,
                        args.model + '_%d'%epoch + '.dat')
                torch.save(model.state_dict(), fname)
                print('saved model at: %s'%fname)
                if test_metrics[-1][0] < best_ssim:
                    best_ssim = test_metrics[-1][0]
                    bname = os.path.join(args.dir_final_test, 'best_model.txt')
                    with open(bname, 'w') as f:
                        f.write(fname)
                        f.close()
                    print('updated best model at: ',bname)

        epoch += 1
    
    #saving data
    #timings
    fname = os.path.join(args.dir_timings, 'all_timings')
    np.save(fname, time_all)

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