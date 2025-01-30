import numpy as np


def large_gfac_matrix(n):
    """
    Creates an n x n matrix with 1 on the main diagonal, 
    -1 on all subdiagonal entries, and 1 in the last column.
    
    Parameters:
        n (int): Size of the matrix (n x n).
        
    Returns:
        numpy.ndarray: The resulting matrix.
    """
    # Initialize an n x n matrix of zeros
    matrix = np.zeros((n, n))
    
    # Set 1 on the main diagonal
    np.fill_diagonal(matrix, 1)
    
    # Set -1 on all subdiagonals
    for k in range(1, n):
        np.fill_diagonal(matrix[k:, :n-k], -1)
    
    # Set 1 in the last column
    matrix[:, -1] = 1
    
    return matrix



def hilbert_matrix(n):
    """Returns the n x n Hilbert matrix.
    """
    return np.fromfunction(lambda i, j: 1 / (i + j + 1), (n, n), dtype=int)



def vandermonde_matrix(n):
    """
    Generate an n x n Vandermonde matrix using n equispaced points in [0, 1].

    Parameters:
    n (int): The size of the matrix (n x n).

    Returns:
    np.ndarray: The generated Vandermonde matrix.
    """
    # Generate n equispaced points in [0, 1]
    points = np.linspace(0, 1, n)
    
    # Create the Vandermonde matrix
    V = np.vander(points, N=n, increasing=True)
    
    return V





