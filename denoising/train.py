import sys
import os
import torch
from torch import optim
import numpy as np
import matplotlib.pyplot as plt

import time as tm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import compute_psnr

def train_epoch(args, train_data, model, optimizer, scheduler, mse_loss):
    for i, (clear, noised) in enumerate(train_data):
        clear = clear.to(args.device)
        noised = noised.to(args.device)
        optimizer.zero_grad()
        denoised_diff, perceptual_loss = model(clear, noised)
        denoised_img = denoised_diff + noised
        loss = perceptual_loss + mse_loss(denoised_img, clear)
        loss.backward()
        optimizer.step()
    scheduler.step()
    return loss.item()


def test_epoch(args, val_data, model, mse_loss):
    model.eval()
    psnr = []
    mse = []
    for i, (clear, noised) in enumerate(val_data):
        start = tm.time()
        res = model.forward_image(noised, args.device, args.test_batch_size)
        psnr.append(compute_psnr(clear, res))
        mse.append(mse_loss(clear, res).item())
        print('Test Iteration time: %.4f'%(tm.time()-start))
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
            test_metrics.append(test_epoch(args, val_data, model, mse_loss))
            print('test done ...')

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

    '''
    #once collected and saved, these things must be analyzed
    #with a separate script
    fname = os.path.join(args.dir_metrics, 'loss_sum.png')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.title.set_text('Loss summary')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.plot(loss_sum)
    plt.savefig(fname)
    plt.close()
    print('saved image at: %s'%fname)
    '''