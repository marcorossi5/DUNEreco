import sys
import os
import torch
from torch import optim
import numpy as np
import matplotlib.pyplot as plt

from model_utils import freeze_weights
from model_utils import MyDataParallel
from model_utils import split_img
from model_utils import recombine_img
from model_utils import plot_crops
from model_utils import plot_ROI_stats
from model_utils import weight_scan

from losses import loss_ssim

from time import time as tm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import compute_psnr

def train_epoch(args, epoch, train_data, model, optimizer, warmup=False):
    print('\n[+] Training')
    model.train()
    for i, (clear, noised) in enumerate(train_data):
        clear = clear.to(args.device)
        noised = noised.to(args.device)
        optimizer.zero_grad()
        loss, out= model(noised, clear, warmup=warmup)
        loss.mean().backward()
        optimizer.step()

        out = out.cpu().detach()
        clear = clear.cpu().detach()

        #plot the last crops chunk
    if args.plot_acts:
        sample = torch.randint(0,
                        out.shape[0],
                       (25,))
        if warmup == 'dn':
            plot_crops(args.dir_testing,
                      out,
                      "act_epoch%d_DN"%epoch,
                      sample)
            plot_crops(args.dir_testing,
                    clear[:,:1],
                    "act_epoch%d_clear"%epoch,
                    sample)
        if warmup == 'roi':
            plot_crops(args.dir_testing,
                       out,
                       "act_epoch%d_DNhits"%epoch,
                       sample)
        
            plot_crops(args.dir_testing,
                       clear[:,1:2],
                       "act_epoch%d_hits"%epoch,
                       sample)
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.view(-1))
    grads = torch.cat(grads).abs().mean()
    print("Grads average: ", grads.item())

    return np.array([loss.mean().item()])

def plot_test_panel(labels, res, fname):
    """
    Plot the inference results
    Params:
        labels: torch.Tensor, shape (w,h)
        res: torch.Tensor, shape (w,h)
        fname: str
    """
    fig = plt.figure()
    gs = fig.add_gridspec(nrows=2, ncols=1, wspace=1.5)

    ax = fig.add_subplot(gs[0])
    z = ax.imshow(labels)
    plt.colorbar(z, ax=ax)
    ax.set_title('Labels')

    ax = fig.add_subplot(gs[1])
    z = ax.imshow(res)
    plt.colorbar(z, ax=ax)
    ax.set_title('Predicted')

    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()

def test_epoch(args, epoch, test_data, model,
               ana=False, warmup=False, labels=None):
    """
    Parameters:
        labels: np.array, all the targets in memory,
                shape (N,C,w,h)
    Outputs:
        np.array: n metrics, shape (2*n)
        torch.Tensor: denoised data, shape (batch,C,W,H)
        float: dry inference time
    """
    print('[+] Testing')
    start = tm()
    model.eval()
    loss = []
    res = [] #inference results

    if warmup == 'dn':
        mse = []
        psnr = []
        ssim = []

    for clear, noisy, norm in test_data:
        if warmup == 'roi':
            target = clear[:,1:2].to(args.device)
        if warmup == 'dn':
            target = clear[:,:1].to(args.device)        
        noisy = noisy.to(args.device)
        norm = norm[0].to(args.device)
        crops, crops_shape, pad = split_img(noisy,model.patch_size)
        loader = torch.split(crops,args.test_batch_size)
        dn = []
        for chunk in loader:
            answer = model(chunk.to(args.device)).data
            dn.append(answer)
        dn = torch.cat(dn).unsqueeze(1)
        dn = recombine_img(dn, crops_shape, pad)
        if warmup == 'roi':
            loss.append(model.xent(target, dn).cpu().item())
        if warmup == 'dn':
            dn = dn * (norm[1]-norm[0]) + norm[0]
            loss.append((model.loss_fn(target, dn)).cpu().item())
            ssim.append(1-loss_ssim()(target, dn).cpu().item())
            mse.append(torch.nn.MSELoss()(target, dn).cpu().item())
            psnr.append(compute_psnr(target, dn))
        res.append(dn.cpu().detach())
    res = torch.cat(res)
    end = tm()
    dry_inf = end-start
    n = len(loss)

    fname = os.path.join(args.dir_testing, f'test_{warmup}_{epoch}')

    if warmup == 'roi':
        #plot_test_panel(labels[0,0], (res[0,0] > args.t).long(),fname)
        plot_test_panel(labels[0,0, 550:700, 5500:], res[0,0, 550:700, 5500:],fname)
        #plot_ROI_stats(args,epoch,labels,res,args.t,ana)
        print('Confusion matrix time:', tm()-end)
        return np.array([np.mean(loss), np.std(loss)/np.sqrt(n)]), res, dry_inf

    if warmup == 'dn':
        plot_test_panel(labels[0,0], res[0,0], fname)
        return np.array([np.mean(loss), np.std(loss)/np.sqrt(n),
                np.mean(ssim), np.std(ssim)/np.sqrt(n),
                np.mean(psnr), np.std(psnr)/np.sqrt(n),
                np.mean(mse), np.std(mse)/np.sqrt(n)]), res, dry_inf


########### main train function
def train(args, train_data, test_data, model, warmup, labels):
    # check if load existing model
    model = freeze_weights(model, warmup)
    model = MyDataParallel(model, device_ids=args.dev_ids)
    model = model.to(args.device)

    if args.load:
        if args.load_path is None:
            #resume a previous training from an epoch
            fname = os.path.join(args.dir_timings, 'timings.npy')
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

            fname = os.path.join(args.dir_saved_models,
                args.model + f'_{warmup}_{args.load_epoch}.dat')
            epoch = args.load_epoch + 1
       
        else:
            #load a trained roi to train new dn
            fname = args.load_path
            epoch = 1
            loss_sum = []
            test_metrics = []
            test_epochs = []
            time_all = []

        print(f'Loading model at {fname}')
        model.load_state_dict(torch.load(fname))
    else:
        epoch = 1
        loss_sum = []
        test_metrics = []
        test_epochs = []
        time_all = []


    best_loss = 1e10
    best_loss_std = 0
    best_model_name = os.path.join(args.dir_saved_models,
                             f'{args.model}_-1.dat')
        
    # initialize optimizer
    lr = args.lr_roi if (warmup=='roi') else args.lr_dn
    optimizer = optim.Adam(list(model.parameters()), lr=lr,
                           amsgrad=args.amsgrad)
    
    # start main loop
    while epoch <= args.epochs:
        time_start = tm()
        # train
        loss = train_epoch(args, epoch, train_data, model,
                          optimizer,warmup=warmup)
        loss_sum.append(loss)

        time_end = tm()-time_start
        time_all.append(time_end)

        if epoch % args.epoch_log == 0 and (not args.scan):
            print("Epoch: %d, Loss: %.5f, time: %.5f"%(epoch,
                                                      loss_sum[-1][0],
                                                      time_end))
        # test
        if epoch % args.epoch_test == 0 and epoch>=args.epoch_test_start:
            print('test start ...')
            test_epochs.append(epoch)
            start = tm()
            x, _, _ = test_epoch(args, epoch, test_data, model,
                              warmup=warmup, labels=labels)
            test_metrics.append(x)
            if not args.scan:
                if warmup == 'roi':
                    print('Test loss: %.5f +- %.5f'%(x[0], x[1]))
                if warmup == 'dn':
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
                fname = os.path.join(args.dir_saved_models,
                         f'{args.model}_{warmup}_{epoch}.dat')
                best_model_name = fname
                bname = os.path.join(args.dir_final_test, 'best_model.txt')
                with open(bname, 'w') as f:
                    f.write(fname)
                    f.close()
                if not args.scan:
                    print('updated best model at: ',bname)
            if args.save:
                fname = os.path.join(args.dir_saved_models,
                         f'{args.model}_{warmup}_{epoch}.dat')
                if not args.scan:
                    print('saved model at: %s'%fname)

            torch.save(model.state_dict(), fname)

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

    #time all
    fname = os.path.join(args.dir_timings, 'timings')
    np.save(fname, time_all)

    return best_loss, best_loss_std, best_model_name
