import numpy as np
import matplotlib.pyplot as plt
import math


def getSphereDepthMap(size=[64, 64], radius=1.0):
    """
    Calculate the positive z-coordinate on the surface of a sphere with the given radius.
    
    Equation:
        x^2 + y^2 + z^2 = radius^2
        
    Given x and y, this function computes:
        z = sqrt(radius^2 - (x^2 + y^2))
        
    Parameters:
        size (list, optional): Dimensions of the output image (default is [64, 64]).
        radius (float, optional): Radius of the sphere (default is 1.0).
    
    Returns:
        float: The positive value of z.
        
    Raises:
        ValueError: If x^2 + y^2 is greater than radius^2.
    """
    x_lin = np.linspace(-1, 1, size[0])
    y_lin = np.linspace(1, -1, size[1])
    z = np.zeros(size)

    # Compute z at (x, y) using sphere formula, i.e., x**2 + y**2 + z**2 = radius**2
    for i, x in enumerate(x_lin):
        for j, y in enumerate(y_lin):
            if x**2 + y**2 > radius**2:
                z[i, j] = np.nan
            else:
                z[i, j] = math.sqrt(radius**2 - x**2 - y**2)
    
    return z

def depthConeDepthMap(size=[64, 64], a=2, b=2, c=2):
    """
    Calculate the positive z-coordinate on the surface of a right circular cone.

    Parameters:
        size (list, optional): Dimensions of the output image (default is [64, 64]).
        a (float, optional): Semi-major axis of the cone (default is 2).
        b (float, optional): Semi-minor axis of the cone (default is 2).
        c (float, optional): Height of the cone (default is 2).

    Returns:
        float: The positive z-value on the cone.
    """
    x_lin = np.linspace(-1, 1, size[0])
    y_lin = np.linspace(1, -1, size[1])
    z = np.zeros(size)

    for i, x in enumerate(x_lin):
        for j, y in enumerate(y_lin):
            z[i, j] = c * math.sqrt((x**2)/(a**2) + (y**2)/(b**2))

    return z

def generateImage(s, alpha, noise=0.0, depthMap=None):
    """
    Computes the dot product between the unit normal at (x, y) on the depthMap 
    and the vector s, raises it to the power alpha, then adds noise.
    
    The normal is computed as:
        p = ∂z/∂x,   q = ∂z/∂y
        n = (-p, -q, 1)/sqrt(1+p^2+q^2)
    
    Parameters:
        s (array-like): A 3-dimensional vector.
        alpha (float): Exponent for the dot product.
        noise (float, optional): Noise added to the result after exponentiation.
        depthMap (np.ndarray): A 2D NumPy array representing the depth map.

    Returns:
        np.adarray: (n dot s)^alpha + noise.
    
    Raises:
        ValueError: If (x^2 + y^2) > radius^2.
    """
    if depthMap is None:
        print('No depth map provided, Generating sphere depth map with raddius 1')
        depthMap = getSphereDepthMap()

    # Compute the gradients
    p = np.gradient(depthMap, axis=0)
    q = np.gradient(depthMap, axis=1)
    
    mask = np.ones(p.shape)
    mask[np.isnan(p)] = 0
    mask[np.isnan(q)] = 0
    
    p[np.isnan(p)] = np.nanmin(p)
    q[np.isnan(q)] = np.nanmin(q)


    # Compute the normal vector
    denom = np.sqrt(1 + p**2 + q**2)

    # Compute the dot product
    dot = (-p * s[0] - q * s[1] + s[2]) * mask / denom
    image = dot ** alpha
    # Normalize image
    image = (image - np.min(image)) / (np.max(image) - np.min(image))

    image *= mask
    # Add noise
    image += noise
 
    return p, q, image 

def generateReflecatance(p, q, s, alpha):
    denom = math.sqrt(1 + p**2 + q**2)
    dot = (-p * s[0] - q * s[1] + s[2]) / denom
    dot = np.clip(dot, 0, 1)
    return dot ** alpha

def generateFullRef(p, q, s, alpha):
    image = np.zeros((p.shape[0], q.shape[1]))
    
    for i in range(p.shape[0]):
        for j in range(q.shape[1]):
            try:
                image[i, j] = generateReflecatance(p[i][j], q[i][j], s, alpha)
            except ValueError:
                # Outside the valid sphere projection, set the pixel value to 0.
                image[i, j] = 0.0
    return image


# Example usage
if __name__ == '__main__':
    surface = generate_smooth_surface(noise=0.1, alpha=1.0)