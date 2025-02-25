### Step 1: Define the Function for the Smooth Surface

We will create a function that generates a smooth surface based on the given parameters. The function will take in arguments for noise and alpha, and it will return a 2D NumPy array representing the surface.

#### Script for Step 1

```python
import numpy as np

def generate_smooth_surface(size=(100, 100), noise=0.1, alpha=1.0):
    """
    Generate a smooth surface using a Gaussian function with added noise.
    
    Parameters:
    - size: Tuple indicating the dimensions of the surface (height, width).
    - noise: Float indicating the amount of noise to add to the surface.
    - alpha: Float controlling the smoothness of the surface.
    
    Returns:
    - surface: 2D NumPy array representing the smooth surface.
    """
    x = np.linspace(-3, 3, size[0])
    y = np.linspace(-3, 3, size[1])
    X, Y = np.meshgrid(x, y)
    
    # Create a smooth surface using a Gaussian function
    surface = np.exp(-(alpha * (X**2 + Y**2)))
    
    # Add noise to the surface
    noise_array = noise * np.random.normal(size=surface.shape)
    surface += noise_array
    
    return surface
```

### Explanation of the Script

1. **Imports**: We import NumPy for numerical operations.
2. **Function Definition**: We define a function `generate_smooth_surface` that takes three parameters: `size`, `noise`, and `alpha`.
3. **Meshgrid Creation**: We create a grid of x and y values using `np.linspace` and `np.meshgrid`. This allows us to evaluate the surface at each point in a 2D space.
4. **Gaussian Function**: We generate a smooth surface using a Gaussian function, which is controlled by the `alpha` parameter. A higher alpha results in a sharper peak.
5. **Adding Noise**: We generate random noise using `np.random.normal` and add it to the surface to create variability.
6. **Return Value**: The function returns the generated surface as a 2D NumPy array.

### Step 2: Visualize the Surface

Next, we will visualize the generated surface using Matplotlib. This will help us see how the parameters affect the smoothness and noise of the surface.

#### Script for Step 2

```python
import matplotlib.pyplot as plt

def plot_surface(surface):
    """
    Plot the smooth surface using Matplotlib.
    
    Parameters:
    - surface: 2D NumPy array representing the smooth surface.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(surface, extent=(-3, 3, -3, 3), origin='lower', cmap='viridis')
    plt.colorbar(label='Surface Value')
    plt.title('Smooth Surface Visualization')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()
```

### Explanation of the Visualization Script

1. **Imports**: We import Matplotlib for plotting.
2. **Function Definition**: We define a function `plot_surface` that takes a 2D NumPy array as input.
3. **Plotting**: We use `plt.imshow` to display the surface. The `extent` parameter sets the limits of the x and y axes, and `origin='lower'` ensures that the (0,0) point is at the bottom-left corner.
4. **Colorbar and Labels**: We add a colorbar for reference and label the axes and title.

### Step 3: Putting It All Together

Now we can create a main function to generate and visualize the smooth surface based on user-defined parameters.

#### Script for Step 3

```python
def main(size=(100, 100), noise=0.1, alpha=1.0):
    surface = generate_smooth_surface(size, noise, alpha)
    plot_surface(surface)

# Example usage
if __name__ == "__main__":
    main(size=(100, 100), noise=0.2, alpha=2.0)
```

### Explanation of the Main Function

1. **Main Function**: We define a `main` function that takes the same parameters as before.
2. **Generate and Plot**: Inside the main function, we call `generate_smooth_surface` to create the surface and then `plot_surface` to visualize it.
3. **Example Usage**: We provide an example usage of the `main` function, which can be modified to test different parameters.

### Conclusion

With these scripts, you can generate and visualize a smooth surface with adjustable noise and alpha values. You can further enhance the functionality by adding more features, such as saving the generated surface to a file or allowing for interactive parameter adjustments.