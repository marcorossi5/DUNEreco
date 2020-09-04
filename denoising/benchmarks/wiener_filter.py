import os
import numpy as np
from numpy.fft import fft2, ifft2
from scipy.signal import gaussian, convolve2d
import matplotlib.pyplot as plt


def wiener_filter(img, kernel, K):
    kernel /= np.sum(kernel)
    dummy = np.copy(img)
    dummy = fft2(dummy)
    kernel = fft2(kernel, s= img.shape)
    kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + K)
    dummy = dummy * kernel
    dummy = np.abs(ifft2(dummy))
    return dummy

def gaussian_kernel(kernel_size = 3):
    h = gaussian(kernel_size, kernel_size / 3).reshape(kernel_size, 1)
    h = np.dot(h, h.transpose())
    h /= np.sum(h)
    return h


if __name__ == '__main__':
    # Load image and convert it to gray scale
    file_name = os.path.join("../datasets/denoising/val", 'planes', 'collection_clear.npy') 
    img = np.load(file_name)[0,0]

    # Blur the image
    file_name = os.path.join("../datasets/denoising/val", 'planes', 'collection_noisy.npy') 
    noisy_img = np.load(file_name)[0,0]

    # Apply Wiener Filter
    kernel = gaussian_kernel(3)
    filtered_img = wiener_filter(noisy_img, kernel, K = 10)

    fig = plt.figure()
    gs = fig.add_gridspec(nrows=2, ncols=1, wspace=3)

    ax = fig.add_subplot(gs[0])
    ax.plot(img[10], lw=0.3)
    #z = ax.imshow(img)
    #plt.colorbar(z, ax=ax)
    ax.set_title('Labels')

    ax = fig.add_subplot(gs[1])
    ax.plot(filtered_img[10], lw=0.3)
    #z = ax.imshow(filtered_img)
    #plt.colorbar(z, ax=ax)
    ax.set_title('Predicted')
        
    plt.savefig('wiener_filter.png', dpi=300, bbox_inches='tight')
    plt.close()

    import torch
    loss = torch.nn.MSELoss()
    mse = loss(torch.Tensor(filtered_img-500), torch.Tensor(img))
    print('MSE clear-wiener filtered: ', mse)
