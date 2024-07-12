import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def lorenz(rho, sigma, beta, initV=[0, 1, 1.05], T=[0, 25], eps=0.000001):
    """
    Function generates the Lorenz attractor trajectories.

    Parameters:
    rho : float
        Rayleigh number.
    sigma : float
        Prandtl number.
    beta : float
        Parameter.
    initV : list, optional
        Initial point [x0, y0, z0]. Default is [0, 1, 1.05].
    T : list, optional
        Time interval [t0, tf]. Default is [0, 25].
    eps : float, optional
        ODE solver precision (absolute and relative tolerance). Default is 0.000001.

    Returns:
    x : array
        Array of x-coordinate values of the Lorenz attractor.
    y : array
        Array of y-coordinate values of the Lorenz attractor.
    z : array
        Array of z-coordinate values of the Lorenz attractor.
    """

    # Define the function F representing the Lorenz system
    def F(t, X):
        dx = np.zeros(3)
        dx[0] = sigma * (X[1] - X[0])
        dx[1] = X[0] * (rho - X[2]) - X[1]
        dx[2] = X[0] * X[1] - beta * X[2]
        return dx

    # Solve the ODE using solve_ivp
    sol = solve_ivp(F, T, initV, method='RK45', atol=eps, rtol=eps)

    # Extract the solution
    t = sol.t
    X = sol.y.T  # X has shape (len(t), 3)

    # Plot the 3D trajectory
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(X[:, 0], X[:, 1], X[:, 2], lw=0.5)
    ax.set_title('Lorenz Attractor')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.grid(True)
    plt.show()

    # Save results to CSV file
    res = np.column_stack((X[:, 0], X[:, 1], X[:, 2]))
    np.savetxt('M3.csv', res, delimiter=',')

    # Return x, y, z arrays
    return X[:, 0], X[:, 1], X[:, 2]

# Example usage:
if __name__ == '__main__':
    rho = 28
    sigma = 10
    beta = 8/3
    x, y, z = lorenz(rho, sigma, beta)