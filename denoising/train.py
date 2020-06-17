import sys
import os
import torch
from torch import optim
import numpy as np
import matplotlib.pyplot as plt

from model_utils import split_img
from model_utils import recombine_img
from model_utils import plot_crops

import time as tm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import compute_psnr

def train_epoch(args, epoch, train_data, model, optimizer, scheduler):
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

def test_epoch(args, epoch, test_data, model):
    model.eval()
    mse = []
    psnr = []
    loss = []
    res = []

    for (clear, noisy) in test_data:        
        crops, crops_shape, pad = split_img(noisy,model.patch_size)
        loader = torch.split(crops,args.test_batch_size)
        dn = []
        for chunk in loader:
            answer = model(chunk.to(args.device)).cpu().data
            dn.append(answer)
        dn = torch.cat(dn)
        dn = recombine_img(dn, crops_shape, pad)
        loss.append(model.loss_fn(clear,dn).cpu().item())
        res.append(dn)
    res = torch.cat(res)

    if args.plot_acts:
        sample = torch.randint(0,
                           res.shape[0],
                           (25,)).cpu().detach().numpy()
        plot_crops(args.dir_testing,
                   res.cpu().detach().numpy()[:,0],
                   "act_epoch%d_DN"%epoch,
                   sample)
        plot_crops(args.dir_testing,
                   clear.cpu().detach().numpy()[:,0],
                   "act_epoch%d_label"%epoch,
                   sample)

    return np.array([np.mean(loss), np.std(loss)/np.sqrt(i+1),
                np.mean(psnr), np.std(psnr)/np.sqrt(i+1),
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
    best_loss = 1e6
    best_model_name

        
    # initialize optimizer
    optimizer=  optim.Adam(list(model.parameters()), lr=args.lr)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer,
                                            lambda x: args.decay_lr**x)

    # start main loop
    time_all = np.zeros(args.epochs)
    
    while epoch <= args.epochs:
        time_start = tm.time()
        # train
        loss = train_epoch(args, epoch, train_data, model,
                          optimizer, scheduler)
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
                                           model))
            
            print('Test loss: %.5f +- %.5f,\
                   psnr: %.5f +- %.5f,\
                   mse: %.5e +- %.5e'%(test_metrics[-1][0],
                                       test_metrics[-1][1],
                                       test_metrics[-1][2],
                                       test_metrics[-1][3],
                                       test_metrics[-1][4],
                                       test_metrics[-1][5]))
            print('Test time: %.4f\n'%(tm.time()-start))

            if test_metrics[-1][0] + test_metrics[-1][1] < best_loss:
                best_loss = test_metrics[-1][0]
                best_loss_std = test_metrics[-1][1]
                bname = os.path.join(args.dir_final_test, 'best_model.txt')
                with open(bname, 'w') as f:
                    f.write(fname)
                    f.close()
                print('updated best model at: ',bname)

        # save model checkpoints or save just best model
        if args.save:
            if epoch % args.epoch_save == 0:
                fname = os.path.join(args.dir_saved_models,
                        args.model + '_%d'%epoch + '.dat')
                torch.save(model.state_dict(), fname)
                print('saved model at: %s'%fname)
                best_model_name = fname
        else:
            if epoch % args.epoch_save == 0:
                fname = os.path.join(args.dir_saved_models,
                        args.model + '.dat')
                torch.save(model.state_dict(), fname)
                print('saved model at: %s'%fname)
                best_model_name = fname

                
        epoch += 1
    
    #saving data
    #timings
    #fname = os.path.join(args.dir_timings, 'all_timings')
    #np.save(fname, time_all)

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