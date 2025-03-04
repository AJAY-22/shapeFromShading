import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from utils import *
from recover import *
from mpl_toolkits.mplot3d import Axes3D

if __name__ == '__main__':
    # Generate the image
    radius = 1.0
    alpha = 1.0
    source = [0, 0, 1]
    noise = 0.0
    sphere = True

    if sphere:
        depthMap = getSphereDepthMap(size=[64, 64], radius=radius)
    else:
        depthMap = getConeDepthMap(a, b, c, size=[64, 64])

    p, q, image = generateImage(source, alpha, noise, depthMap)
    
    # image[np.isnan(image)] = np.nanmin(image)
    plt.imshow(image, cmap='gray')
    plt.title('Image')
    plt.show()
    # quit()

    if sphere:
        os.makedirs(f'output/sphere/r={radius}', exist_ok=True)
        outputDir = f'output/sphere/r={radius}'
    else:
        os.makedirs(f'output/cone/r={radius}', exist_ok=True)
        outputDir = f'output/cone/r={radius}'

    # np.save(os.path.join(outputDir, 'image.npy'), image)
    # np.save(os.path.join(outputDir, 'p_array_.npy'), p_array)
    # np.save(os.path.join(outputDir, 'q_array.npy'), q_array)

    # Save the image as a PNG file
    obj = 'sphere' if sphere else 'cone'
    # plt.imsave(os.path.join(outputDir, f'{obj}_{source}.png'), image, cmap='gray')
    # plt.imsave(os.path.join(outputDir, f'{obj}_{source}_z.png'), depthMap, cmap='gray')
    

    # print(image)
    # # Display the image
    # plt.imshow(image, cmap='gray')
    # plt.title(f'{obj} Image')
    # plt.axis('off')
    # plt.show()

    # # Display the z values
    depthMap[np.isnan(depthMap)] = np.nanmin(depthMap)
    # plt.imshow(depthMap, cmap='gray')
    # plt.title('Z Values')
    # plt.axis('off')
    # plt.show()

    # ---------- Shape-from-Shading Recovery ----------
    # Here, we treat the generated image as our observed image E.
    # Set regularization parameter lam (e.g., 0.001)
    lam = 7000
    # Recover surface gradients and depth from E using our recover_surface function.
    # p_rec, q_rec, z_rec, r_rec = recover_surface_iterative(image, source, alpha, lam, max_iter=10, tol=1e-6, p0=None, q0=None)
    rec_p, rec_q, rec_z = recoverdepth(image, source, alpha, lam, max_iter=500, smoothingFunc=gaussian, size=[64, 64])

    # Save the recovered components
    # np.save(os.path.join(outputDir, `'p_recovered.npy'), p_rec)
    # np.save(os.path.join(outputDir, 'q_recovered.npy'), q_rec)
    # np.save(os.path.join(outputDir, 'z_recovered.npy'), z_rec)
    
    # # Display the recovered depth (z) map
    # plt.imshow(z_rec, cmap='gray')
    # plt.title('Recovered Z Values')
    # plt.axis('off')
    # plt.show()


    # Plot all 3: actual image, z values and recovered z values.
    fig, axs = plt.subplots(2, 3, figsize=(10, 5))

    axs[0][0].imshow(image, cmap='gray')
    axs[0][0].set_title('Actual Image')
    axs[0][0].axis('off')

    ax = fig.add_subplot(232, projection='3d')
    X, Y = np.meshgrid(np.linspace(-1, 1, 64), np.linspace(-1, 1, 64))
    ax.plot_surface(X, Y, depthMap, cmap='gray', edgecolor='none')
    ax.set_title('Depth Map')
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")

    ax = fig.add_subplot(233, projection='3d')
    X, Y = np.meshgrid(np.linspace(-1, 1, 64), np.linspace(-1, 1, 64))
    ax.plot_surface(X, Y, rec_z, cmap='gray', edgecolor='none')
    ax.set_title('Reconstructed Depth Map')
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")


    axs[1][0].imshow(rec_p, cmap='gray')
    axs[1][0].set_title('Recovered P Values')
    axs[1][0].axis('off')

    axs[1][1].imshow(rec_p, cmap='gray')
    axs[1][1].set_title('Recovered P Values')
    axs[1][1].axis('off')

    axs[1][2].imshow(rec_q, cmap='gray')
    axs[1][2].set_title('Recovered Q Values')
    axs[1][2].axis('off')

    plt.tight_layout()
    plt.show()





