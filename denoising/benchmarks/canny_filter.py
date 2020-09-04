import os
import numpy as np
from skimage.feature import canny
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # Load target images
    file_name = os.path.join("../datasets/denoising/test",
                             'planes', 'collection_clear.npy') 
    img = np.load(file_name)

    # Load noisy images
    file_name = os.path.join("../datasets/denoising/test",
                             'planes', 'collection_noisy.npy') 
    noisy_img = np.load(file_name)

    filtered_img = []

    # Apply Canny Filter
    for i in noisy_img:
        im = canny(i[0]).astype(float)
        filtered_img.append(im)
    filtered_img = np.stack(filtered_img)[:, None]

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
