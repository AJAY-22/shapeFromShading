import numpy as np
import math
from scipy.optimize import minimize
from tqdm import tqdm
from main import generate_full_image
from utils import generateFullRef

def reflectance(p, q, s, alpha):
    """
    Compute the reflectance R(x,y) for the given p and q.
    
    The surface normal is:
        n = [-p, -q, 1] / sqrt(1 + p^2 + q^2)
        
    and the reflectance map is defined as:
        R(x,y) = (n dot s)^alpha
    """
    n = np.array([-p, -q, np.ones_like(p)]) / np.sqrt(1 + p**2 + q**2)
    R = np.tensordot(n, s, axes=([0], [0]))
    # Clamp R to be non-negative
    R = np.clip(R, 0, 1)
    return R**alpha

def cost_function(v, E, s, alpha, lam, shape):
    """
    The cost function to minimize.
    
    v is a flattened vector containing p and q (each of shape m x n).
    
    Cost = MSE(E, R(p,q)) + lam * regularizer(p,q)
         = mean((E - R)^2) + lam * mean(p^2 + q^2)
    """
    radius = 1.0
    m, n = shape
    # Extract p and q from the vector v
    p = v[:m*n].reshape((m, n))
    q = v[m*n:].reshape((m, n))

    R = generateFullRef(p, q, s, alpha)
    mse = np.mean((E - R) ** 2)

    # Create a coordinate grid assuming x and y vary uniformly in [-radius, radius]
    x = np.linspace(-radius, radius, m)
    y = np.linspace(radius, -radius, n)
    # Use 'ij' indexing so X[i,j] corresponds to x_i and Y[i,j] to y_j:
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Compute r = sqrt(radius^2 - (x^2+y^2)).
    # (Take care to avoid division by zero outside the valid domain.)
    r = np.sqrt(np.clip(radius**2 - (X**2 + Y**2), 1e-8, None))
    
    # Closed-form second derivatives of z (which is the primitive of p, q)
    d2z_dx2 = -1/r - (X**2)/(r**3)
    d2z_dy2 = -1/r - (Y**2)/(r**3)
    d2z_dxdy = - (X*Y)/(r**3)
    
    # Regularizer: square of the second derivatives (note the cross derivative appears twice)
    reg = lam * (np.mean(d2z_dx2**2) + 2*np.mean(d2z_dxdy**2) + np.mean(d2z_dy2**2))
    
    return mse + reg

def recover_surface(E, s, alpha, lam, p0=None, q0=None):
    """
    Recover p(x,y) and q(x,y) from the given image E(x,y) by minimizing: 
    
        MSE(E, R(x,y,p,q)) + lam*(regularizer)
        
    where R(x,y) = (n(x,y) dot s)^alpha and
          n(x,y) = [-p(x,y), -q(x,y), 1] / sqrt(1+p(x,y)^2+q(x,y)^2)
    
    After recovering p and q, z(x,y) is obtained by a simple integration.
    
    Parameters:
        E (np.ndarray): Observed image, E(x,y).
        s (array-like): Light source direction (3D vector).
        alpha (float): Exponent in the reflectance model.
        lam (float): Regularization weight on p and q.
        p0 (np.ndarray, optional): Initial guess for p (default zeros).
        q0 (np.ndarray, optional): Initial guess for q (default zeros).
    
    Returns:
        tuple: (p_opt, q_opt, z) where z is integrated from p and q.
    """
    m, n = E.shape
    # Initial guesses (if not provided, use zeros)
    if p0 is None:
        # p0 = np.random.uniform(-1, 1, size=(m, n))
        p0 = np.zeros((m, n))
    if q0 is None:
        # q0 = np.random.uniform(-1, 1, size=(m, n))
        q0 = np.zeros((m, n))
    # Flatten and concatenate p and q as optimization variables.
    v0 = np.concatenate([p0.flatten(), q0.flatten()])
    result = minimize(cost_function, v0, args=(E, s, alpha, lam, E.shape),
                      method='L-BFGS-B')
    
    v_opt = result.x
    p_opt = v_opt[:m*n].reshape((m, n))
    q_opt = v_opt[m*n:].reshape((m, n))
    
    # Recover z from p and q via simple integration.
    # Here, we assume unit spacing in both x and y directions.
    z = np.zeros(E.shape)
    dx = 1.0
    dy = 1.0
    # A simple integration: starting from z[0,0]=0, integrate along rows and columns.
    for i in range(1, m):
        z[i, 0] = z[i-1, 0] + p_opt[i-1, 0] * dx
    for j in range(1, n):
        z[0, j] = z[0, j-1] + q_opt[0, j-1] * dy
    for i in range(1, m):
        for j in range(1, n):
            # Combine row and column integration (this is a crude approximation)
            z[i, j] = (z[i-1, j] + z[i, j-1]) / 2
    return p_opt, q_opt, z




def recover_surface_iterative(E, s, alpha, lam, max_iter=100, tol=1e-6, p0=None, q0=None):
    """
    Iteratively recover p(x,y) and q(x,y) (surface gradients) from
    an observed image E using the update rule:
    
      p₍ᵢⱼ₎ⁿ⁺¹ = p₍ᵢⱼ₎ⁿ - (1/lam) * (E₍ᵢⱼ₎ - R(p,q)₍ᵢⱼ₎) * (∂R/∂p)₍ᵢⱼ₎
      q₍ᵢⱼ₎ⁿ⁺¹ = q₍ᵢⱼ₎ⁿ - (1/lam) * (E₍ᵢⱼ₎ - R(p,q)₍ᵢⱼ₎) * (∂R/∂q)₍ᵢⱼ₎
    
    where the reflectance is computed as:
        R(p,q) = (n · s)^alpha
    with n = [-p, -q, 1] / sqrt(1+p^2+q^2).
    
    Parameters:
        E (np.ndarray): Observed image of shape (m, n).
        s (array-like): Light source direction (3D vector, [s0, s1, s2]).
        alpha (float): Exponent in the reflectance model.
        lam (float): Step size (or learning rate) in the update.
        max_iter (int): Maximum number of iterations.
        tol (float): Tolerance for convergence.
        p0, q0 (np.ndarray, optional): Initial guesses for p and q. If None, random initialization is used.
    
    Returns:
        tuple: (p, q, z) where p,q are recovered gradients and z is computed via integration.
    """
    m, n = E.shape
    # Initialize p and q randomly if not provided
    if p0 is None:
        p = np.random.uniform(-1, 1, size=(m, n))
        # p = np.zeros((m, n))
    else:
        p = p0.copy()
    if q0 is None:
        q = np.random.uniform(1, -1, size=(m, n))
        # q = np.zeros((m, n))
    else:
        q = q0.copy()
    
    s = np.array(s)  # Ensure s is a NumPy array.
    s0, s1, s2 = s[0], s[1], s[2]
    
    for iteration in tqdm(range(max_iter)):
        # R = generateFullRef(p, q, s, alpha)
        R = reflectance(p, q, s, alpha)
        D = np.sqrt(1 + p**2 + q**2)
        dot = (-p*s0 - q*s1 + s2) / D
        # Compute derivative of dot (n·s) with respect to p and q.
        # For each pixel:
        # d(dot)/dp = (-s0*D^2 - (-p*s0 - q*s1 + s2)* (p))/D^3
        #            = (-s0*(1+p^2+q^2) + p*s0 + q*s1 - p*s2) / D^3
        # Simplify numerator: -s0 - s0*q^2 + q*s1 - p*s2.
        d_dot_dp = (-s0*(1 + q**2) + q*s1 - p*s2) / (D**3)
        # Similarly, derivative with respect to q:
        d_dot_dq = (-s1*(1 + p**2) + p*s0 - q*s2) / (D**3)
        
        # Now, by chain rule, compute derivatives of R wrt p and q:
        # ∂R/∂p = α * (dot)^(α-1) * d(dot)/dp
        # ∂R/∂q = α * (dot)^(α-1) * d(dot)/dq
        # (Note: When dot is negative, it is clamped to 0 in R; we assume illumination such that dot >=0)
        dR_dp = alpha * (np.clip(dot, 0, None))**(alpha - 1) * d_dot_dp
        dR_dq = alpha * (np.clip(dot, 0, None))**(alpha - 1) * d_dot_dq
        
        # Compute the update: error = (E - R)
        error = E - R
        
        # Update p and q elementwise based on given rule.
        # print(dR_dp)
        # print('================================')
        # print(dR_dq)
        new_p = p - (1/lam) * error * dR_dp
        new_q = q - (1/lam) * error * dR_dq
        
        # Check convergence (using the max absolute change over all pixels)
        if np.max(np.abs(new_p - p)) < tol and np.max(np.abs(new_q - q)) < tol:
            p, q = new_p, new_q
            print("Converged at iteration", iteration)
            break
        
        p, q = new_p, new_q
    
    # After p and q are recovered, integrate them to obtain z.
    # Simple integration assuming unit spacing.
    z = np.zeros(E.shape)
    dx = 1.0
    dy = 1.0
    for i in range(1, m):
        z[i, 0] = z[i-1, 0] + p[i-1, 0] * dx
    for j in range(1, n):
        z[0, j] = z[0, j-1] + q[0, j-1] * dy
    for i in range(1, m):
        for j in range(1, n):
            z[i, j] = (z[i-1, j] + z[i, j-1]) / 2

    return p, q, z, R


