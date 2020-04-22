import sys
import os
import torch
from torch import optim
import numpy as np
import matplotlib.pyplot as plt

from model_utils import split_img
from model_utils import recombine_img

import time as tm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import compute_psnr

def train_epoch(args, train_data, model, optimizer, scheduler, mse_loss):
    model.train()
    for i, (clear, noised) in enumerate(train_data):
        clear = clear.to(args.device)
        noised = noised.to(args.device)
        optimizer.zero_grad()
        #denoised_diff, perceptual_loss = model(clear, noised)
        #denoised_img = model.act(denoised_diff + noised)
        #loss = perceptual_loss + mse_loss(denoised_img, clear)
        denoised_img, loss = model(noised, clear)
        #c = model(clear)
        #n = model(noised)
        #loss = mse_loss(c[0], n[0]) + mse_loss(c[1], n[1])\
        #       + mse_loss(c[2], n[2]) + mse_loss(clear, n[3])
        loss.sum().backward()
        optimizer.step()
    scheduler.step()
    return loss.sum().item()


def test_epoch(args, epoch, val_data, model, mse_loss):
    model.eval()
    psnr = []
    mse = []
    res = [[],[]]
    labels = [[],[]]
    legend = ['collection', 'readout']

    p_x, p_y = model.patch_size
    split_size = 256

    for i, data in enumerate(val_data):
        for (clear, noised) in data:
            labels[i] += [clear]
            
            crops, crops_shape, pad = split_img(noised, (p_x,p_y))
            loader = torch.split(crops, split_size)
            dn = []
            for chunk in loader:
                answer = model(chunk.to(device)).cpu().data
                dn.append(answer)

            dn = torch.cat(dn)

            dn = recombine_img(dn, crops_shape, pad)

            res[i] += [dn]


            #res[i] += [model.forward_image(noised,
            #                               args.device,
            #                              args.test_batch_size)]
            psnr.append(compute_psnr(clear, res[i][-1]))
            mse.append(mse_loss(clear, res[i][-1]).item())
        labels[i] = np.concatenate(labels[i])[:,0]
        res[i] = np.concatenate(res[i])[:,0]
    
    #saving one example
    '''
    fname = os.path.join(args.dir_testing, 'test_at_%d.png'%epoch)
    fig = plt.figure(figsize=(20,25))
    plt.suptitle('Test epoch: %d denoising example'%epoch)
    ax = fig.add_subplot(311)
    ax.title.set_text('Noised image')
    z = ax.imshow(noised[0,0])
    fig.colorbar(z, ax=ax)

    ax = fig.add_subplot(312)
    ax.title.set_text('Clear image')
    z = ax.imshow(clear[0,0])
    fig.colorbar(z, ax=ax)

    ax = fig.add_subplot(313)
    ax.title.set_text('Denoised image')
    z = ax.imshow(res[-1][0,0])
    fig.colorbar(z, ax=ax)
    plt.savefig(fname)
    plt.close()
    '''
    
    #saving diffs and hists
    diff = [np.abs(res[i] - labels[i]) for i in range(len(res))]

    fname = os.path.join(args.dir_testing, 'residuals_at_%d.png'%epoch)
    fig = plt.figure(figsize=(20,25))
    plt.suptitle('Final denoising test')

    ax = fig.add_subplot(311)
    ax.title.set_text('Sample of Clear image')
    z = ax.imshow(labels[0][0])
    fig.colorbar(z, ax=ax)

    ax = fig.add_subplot(312)
    ax.title.set_text('Sample of |Denoised - Clear|')
    z = ax.imshow(diff[0][0])
    fig.colorbar(z, ax=ax)

    ax = fig.add_subplot(325)
    ax.hist([i[0].flatten() for i in diff], 100,
             stacked=True, label=legend,
             density=True, histtype='step')
    ax.set_yscale('log')
    ax.legend()
    ax.title.set_text('Sample of histogram of |Diff|')

    ax = fig.add_subplot(326)
    ax.hist([i.flatten() for i in diff], 100,
             stacked=True, label=legend,
             density=True, histtype='step')
    ax.set_yscale('log')
    ax.legend()
    ax.title.set_text('Histogram of all |Diff|')

    plt.savefig(fname)
    plt.close()

    return np.array([np.mean(psnr), np.std(psnr)/np.sqrt(i+1),
                np.mean(mse), np.std(mse)/np.sqrt(i+1)])

#this has to be done once best model is selected
def final_test():
    pass

########### main train function
def train(args, train_data, val_data, model):
    # check if load existing model
    if args.load:
        fname = os.path.join(args.dir_saved_models,
            args.model + '_%d'%args.load_epoch + '.dat')
        model.load_state_dict(torch.load(fname))

        args.lr = 5e-4
        epoch = args.load_epoch
        print('model loaded!, lr: {}'.format(args.lr))
    else:
        epoch = 1

    # initialize optimizer
    optimizer=  optim.Adam(list(model.parameters()), lr=args.lr)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer,
                                                 lambda x: args.decay_lr**x)

    # start main loop
    time_all = np.zeros(args.epochs)
    mse_loss = torch.nn.MSELoss()
    loss_sum = []
    test_metrics = []
    test_epochs = []
    while epoch<=args.epochs:
        time_start = tm.time()
        # train
        loss = train_epoch(args,train_data, model,
                          optimizer, scheduler, mse_loss)
        loss_sum.append(loss)

        time_end = tm.time()
        time_all[epoch - 1] = time_end - time_start
        if epoch % args.epoch_log == 0:
            print("Epoch: %d, Loss: %.5f, time: %.5f"%(epoch,
                                                      loss_sum[-1],
                                                      time_all[epoch - 1]))
        # test
        if epoch % args.epoch_test == 0 and epoch>=args.epoch_test_start:
            print('test start ...')
            test_epochs.append(epoch)
            start = tm.time()
            test_metrics.append(test_epoch(args, epoch, val_data,
                                           model, mse_loss))
            print('Test psnr: %.5f +- %.5f, mse: %.5f +- %.5f'%(test_metrics[-1][0],
                                                                test_metrics[-1][1],
                                                                test_metrics[-1][2],
                                                                test_metrics[-1][3]))
            print('Test time: %.4f\n'%(tm.time()-start))

        # save model checkpoint
        if args.save:
            if epoch % args.epoch_save == 0:
                fname = os.path.join(args.dir_saved_models,
                        args.model + '_%d'%epoch + '.dat')
                torch.save(model.state_dict(), fname)
                bname = os.path.join(args.dir_final_test, 'best_model.txt')
                with open(bname, 'w') as f:
                    f.write(fname)
                    f.close()
                print('updated best model at: ',bname)
                print('saved model at: %s'%fname)
        epoch += 1
    
    #saving data
    #timings
    fname = os.path.join(args.dir_timings, 'all_timings')
    np.save(fname, time_all)

    #loss_sum
    fname = os.path.join(args.dir_metrics, 'loss_sum')
    np.save(fname, loss_sum)
    
    #test_epochs
    fname = os.path.join(args.dir_metrics, 'test_epochs')
    np.save(fname, test_epochs)

    #test metrics
    test_metrics = np.stack(test_metrics,1)
    fname = os.path.join(args.dir_metrics, 'test_metrics')
    np.save(fname, test_metrics)