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
    source = [0, 100, 0]
    noise = 0.0
    sphere = True
    z, image, p_array, q_array = generate_full_image((64, 64), source, alpha, noise, radius=radius, sphere=sphere)
    minn = np.min(image)
    image[z == 0] = minn

    # Normalize values in image to [0, 1]
    image = (image - image.min()) / (image.max() - image.min())
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
    
    # Display the image
    # plt.imshow(image, cmap='gray')
    # plt.title(f'{obj} Image')
    # plt.axis('off')
    # plt.show()

    # Display the z values
    # plt.imshow(z, cmap='gray')
    # plt.title('Z Values')
    # plt.axis('off')
    # plt.show()

    # ---------- Shape-from-Shading Recovery ----------
    # Here, we treat the generated image as our observed image E.
    # Set regularization parameter lam (e.g., 0.001)
    lam = 100.0
    # Recover surface gradients and depth from E using our recover_surface function.
    p_rec, q_rec, z_rec, r_rec = recover_surface_iterative(image, source, alpha, lam, max_iter=5, tol=1e-6, p0=None, q0=None)
    
    # Save the recovered components
    np.save(os.path.join(outputDir, 'p_recovered.npy'), p_rec)
    np.save(os.path.join(outputDir, 'q_recovered.npy'), q_rec)
    np.save(os.path.join(outputDir, 'z_recovered.npy'), z_rec)
    
    # # Display the recovered depth (z) map
    # plt.imshow(z_rec, cmap='gray')
    # plt.title('Recovered Z Values')
    # plt.axis('off')
    # plt.show()


    # Plot all 3: actual image, z values and recovered z values.
    fig, axs = plt.subplots(2, 2, figsize=(10, 5))

    axs[0][0].imshow(image, cmap='gray')
    axs[0][0].set_title('Actual Image')
    axs[0][0].axis('off')

    axs[0][1].imshow(z, cmap='gray')
    axs[0][1].set_title('Z Values')
    axs[0][1].axis('off')

    axs[1][0].imshow(z_rec, cmap='gray')
    axs[1][0].set_title('Recovered Z Values')
    axs[1][0].axis('off')


    axs[1][1].imshow(r_rec, cmap='gray')
    axs[1][1].set_title('Recovered R Values')
    axs[1][1].axis('off')

    plt.tight_layout()
    plt.show()

