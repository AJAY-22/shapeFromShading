import numpy as np
import matplotlib.pyplot as plt
import math


def depth_z(x, y, radius=1.0):
    """
    Calculate the positive z-coordinate on the surface of a sphere with the given radius.
    
    Equation:
        x^2 + y^2 + z^2 = radius^2
        
    Given x and y, this function computes:
        z = sqrt(radius^2 - (x^2 + y^2))
        
    Parameters:
        x (float): x-coordinate.
        y (float): y-coordinate.
        radius (float, optional): Radius of the sphere (default is 1.0).
    
    Returns:
        float: The positive value of z.
        
    Raises:
        ValueError: If x^2 + y^2 is greater than radius^2.
    """
    # print(x, y)
    if x**2 + y**2 > radius**2:
        raise ValueError("x^2 + y^2 cannot be greater than radius^2 for a valid point on the sphere")
    return math.sqrt(radius**2 - (x**2 + y**2))

def depth_z_cone(x, y, slope=1.0):
    """
    Calculate the positive z-coordinate on the surface of a right circular cone.

    Cone Equation:
        x^2 + y^2 = (z / slope)^2
    Solving for z we get:
        z = slope * sqrt(x^2 + y^2)

    Parameters:
        x (float): x-coordinate.
        y (float): y-coordinate.
        slope (float, optional): Controls the steepness of the cone (default is 1.0).

    Returns:
        float: The positive z-value on the cone.
    """
    a = 2
    b = 2
    c = 2
    return -1 * c * math.sqrt((x**2)/(a**2) + (y**2)/(b**2))

def generateImage(x, y, s, alpha, noise=0.0, h=1e-5, radius=1.0, sphere=True):
    """
    Computes the dot product between the unit normal at (x, y) on the sphere
    (defined via depth_z) and the vector s, raises it to the power alpha, then adds noise.
    
    The surface is defined as z = depth_z(x, y) where:
        depth_z(x, y) = sqrt(radius^2 - x^2 - y^2)
    
    The normal is computed as:
        p = ∂z/∂x,   q = ∂z/∂y
        n = (-p, -q, 1)/sqrt(1+p^2+q^2)
    
    Parameters:
        x (float): x-coordinate.
        y (float): y-coordinate.
        s (array-like): A 3-dimensional vector.
        alpha (float): Exponent for the dot product.
        noise (float, optional): Noise added to the result after exponentiation.
        h (float, optional): Step size for finite difference. Default is 1e-5.
        radius (float, optional): Radius of the sphere used in depth_z. Default is 1.0.
    
    Returns:
        float: (n dot s)^alpha + noise.
    
    Raises:
        ValueError: If (x^2 + y^2) > radius^2.
    """
    if sphere:
        # Compute z at (x, y) using depth_z function (must be defined elsewhere in utils.py)
        z = depth_z(x, y, radius)
        p = - x/(math.sqrt(radius**2 - x**2 - y**2))
        q = - y/(math.sqrt(radius**2 - x**2 - y**2))
    else:
        z = depth_z_cone(x, y)
        
        # Approximate partial derivatives using central differences
        # p = ∂z/∂x
        z_plus_h = depth_z_cone(x + h, y, radius)
        z_minus_h = depth_z_cone(x - h, y, radius)
        p = (z_plus_h - z_minus_h) / (2 * h)
        
        # q = ∂z/∂y
        z_plus_h = depth_z_cone(x, y + h, radius)
        z_minus_h = depth_z_cone(x, y - h, radius)
        q = (z_plus_h - z_minus_h) / (2 * h)
    
    # Compute the unit normal vector n = (-p, -q, 1) / sqrt(1 + p^2 + q^2)
    denom = math.sqrt(1 + p**2 + q**2)
    n = np.array([p, q, 1.0]) / denom
    
    # Compute dot product of n and s
    s = np.array(s)
    dot = np.dot(n, s)
    
    # Return (dot**alpha) plus noise
    return z, (dot ** alpha) + noise, p, q

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

# def generate_smooth_surface(noise=0.1, alpha=1.0):
#     """
#     Generate a smooth surface based on the given parameters.

#     Parameters:
#     - size: Tuple indicating the dimensions of the surface (height, width).
#     - noise: Float indicating the amount of noise to add to the surface.
#     - alpha: Float controlling the smoothness of the surface.

#     Returns:
#     - surface: 2D NumPy array representing the smooth surface.
#     """
#     # Create a grid of points
#     x = np.linspace(-3, 3, size[0])
#     y = np.linspace(-3, 3, size[1])
#     X, Y = np.meshgrid(x, y)

#     # Generate a smooth function (e.g., Gaussian)
#     Z = depth_z(x, y)

#     # Add noise to the surface
#     noise_array = noise * np.random.normal(size=Z.shape)
#     surface = Z + noise_array

#     return surface

# Example usage
if __name__ == '__main__':
    surface = generate_smooth_surface(noise=0.1, alpha=1.0)