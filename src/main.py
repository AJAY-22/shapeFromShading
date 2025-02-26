import os

import numpy as np
import matplotlib.pyplot as plt
from utils import generateImage
from recover import *

def generate_full_image(size=(64, 64), s=[0, 0, 1], alpha=1.0, noise=0.0, radius=1.0, sphere=True):
    """
    Generate a 2D image of dimensions `size` (default 64x64) where each pixel
    is computed using the generateImage function. x and y values are linearly
    sampled from -radius to radius. If a point lies outside the valid domain on
    the sphere (i.e. x^2+y^2 > radius^2), that pixel is set to 0.
    
    Parameters:
        size (tuple): Dimensions of the output image, as (rows, columns). Default is (64, 64).
        s (array-like): A 3-dimensional vector used in the dot product with the surface normal.
                        Default is [0, 0, 1].
        alpha (float): Exponent used in generateImage.
        noise (float): Noise to be added after exponentiation in generateImage.
        h (float): Step-size for numerical differentiation in generateImage.
        radius (float): Radius of the sphere defined in depth_z; also used to define the valid (x, y) domain.
    
    Returns:
        np.ndarray: A 2D NumPy array of shape `size` representing the generated image.
    """
    image = np.zeros(size)
    p_array = np.zeros(size)
    q_array = np.zeros(size)
    z = np.zeros(size)

    x_lin = np.linspace(-1, 1, size[0])
    y_lin = np.linspace(1, -1, size[1])
    
    for i, x in enumerate(x_lin):
        for j, y in enumerate(y_lin):
            try:
                z[i, j], image[i, j], p_array[i, j], q_array[i, j] = generateImage(x, y, s, alpha, noise, radius, sphere)
            except ValueError:
                # Outside the valid sphere projection, set the pixel value to 0.
                image[i, j] = 0.0
    return z, image, p_array, q_array


if __name__ == '__main__':
    # Generate the image
    radius = 1.0
    alpha = 1.0
    source = [0, 0, 1]
    noise = 0.0
    sphere = True
    z, image, p_array, q_array = generate_full_image((64, 64), source, alpha, noise, radius=radius, sphere=sphere)

    p = np.zeros(z.shape)
    q = np.zeros(z.shape)
    for j in range(0, z.shape[1]):
        p[0, j] = z[0, j]
    for i in range(0, z.shape[0]):
        q[i, 0] = z[i, 0]
    for i in range(1, z.shape[0]):
        for j in range(0, z.shape[1]):
            p[i, j] = z[i, j] - z[i-1, j]
    for i in range(0, z.shape[0]):
        for j in range(1, z.shape[1]):
            q[i, j] = z[i, j] - z[i, j-1]
    p_array = p
    q_array = q
    # Display the image
    minn = np.min(image)
    image[z == 0] = minn

    # Normalize values in image to [0, 1]
    # image = (image - image.min()) / (image.max() - image.min())
    # quit()
    if sphere:
        os.makedirs(f'output/sphere/r={radius}', exist_ok=True)
        outputDir = f'output/sphere/r={radius}'
    else:
        os.makedirs(f'output/cone/r={radius}', exist_ok=True)
        outputDir = f'output/cone/r={radius}'

    np.save(os.path.join(outputDir, 'image.npy'), image)
    np.save(os.path.join(outputDir, 'p_array_.npy'), p_array)
    np.save(os.path.join(outputDir, 'q_array.npy'), q_array)

    # Save the image as a PNG file
    obj = 'sphere' if sphere else 'cone'
    plt.imsave(os.path.join(outputDir, f'{obj}_{source}.png'), image, cmap='gray')
    plt.imsave(os.path.join(outputDir, f'{obj}_{source}_z.png'), z, cmap='gray')
    

    # ---------- Shape-from-Shading Recovery ----------
    # Here, we treat the generated image as our observed image E.
    # Set regularization parameter lam (e.g., 0.001)
    lam = 100.0
    # Recover surface gradients and depth from E using our recover_surface function.
    p_rec, q_rec, z_rec, r_rec = recover_surface_iterative(image, source, alpha, lam, max_iter=10000, tol=1e-20, p0=None, q0=None)
    
    # Save the recovered components
    np.save(os.path.join(outputDir, 'p_recovered.npy'), p_rec)
    np.save(os.path.join(outputDir, 'q_recovered.npy'), q_rec)
    np.save(os.path.join(outputDir, 'z_recovered.npy'), z_rec)
    
    # # Display the recovered depth (z) map
    # plt.imshow(z_rec, cmap='gray')
    # plt.title('Recovered Z Values')
    # plt.axis('off')
    # plt.show()



    z1 = np.zeros(image.shape)
    dx = 1.0
    dy = 1.0
    for i in range(1, image.shape[0]):
        z1[i, 0] = z1[i-1, 0] + p_array[i-1, 0] * dx
    for j in range(1, image.shape[1]):
        z1[0, j] = z1[0, j-1] + q_array[0, j-1] * dy
    for i in range(1, image.shape[0]):
        for j in range(1, image.shape[1]):
            z11 = z1[i-1, j] + p_array[i, j] * dx
            z12 = z1[i, j-1] + q_array[i, j] * dy
            z1[i,j] = (z11+z12)/2
    # Plot all 3: actual image, z values and recovered z values.

    fig, axs = plt.subplots(2, 5, figsize=(10, 5))
    axs[0][0].imshow(image, cmap='gray')
    axs[0][0].set_title('Actual Image')
    axs[0][0].axis('off')

    axs[0][1].imshow(z, cmap='gray')
    axs[0][1].set_title('Z Values')
    axs[0][1].axis('off')


    axs[0][2].imshow(p_array, cmap='gray')
    axs[0][2].set_title('actual p')
    axs[0][2].axis('off')


    axs[0][3].imshow(q_array, cmap='gray')
    axs[0][3].set_title('actual q')
    axs[0][3].axis('off')

    
    axs[0][4].imshow(z1, cmap='gray')
    axs[0][4].set_title('z recontructed from actual p and actual q')
    axs[0][4].axis('off')

    axs[1][0].imshow(p_rec, cmap='gray')
    axs[1][0].set_title('Recovered p Values')
    axs[1][0].axis('off')


    axs[1][1].imshow(q_rec, cmap='gray')
    axs[1][1].set_title('Recovered q Values')
    axs[1][1].axis('off')


    axs[1][2].imshow(z_rec, cmap='gray')
    axs[1][2].set_title('Recovered Z Values')
    axs[1][2].axis('off')


    axs[1][3].imshow(r_rec, cmap='gray')
    axs[1][3].set_title('Recovered R Values')
    axs[1][3].axis('off')


    plt.tight_layout()
    plt.show()

