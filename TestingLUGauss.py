import unittest
import numpy as np

def Lu_Gauss(A):
    """
    Melakukan dekomposisi LU pada matriks A.
    
    Parameters:
        A (numpy.ndarray): Matriks koefisien.
        
    Returns:
        numpy.ndarray: Matriks segitiga bawah L.
        numpy.ndarray: Matriks segitiga atas U.
    """
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    
    for i in range(n):
        L[i][i] = 1
        
        for j in range(i, n):
            U[i][j] = A[i][j]
            
            for k in range(i):
                U[i][j] -= L[i][k] * U[k][j]
        
        for j in range(i+1, n):
            L[j][i] = A[j][i]
            
            for k in range(i):
                L[j][i] -= L[j][k] * U[k][i]
            L[j][i] /= U[i][i]
    
    return L, U

def solusi_sistem_persamaan_linear(A, b):
    """
    Menyelesaikan sistem persamaan linear Ax = b menggunakan metode dekomposisi LU.
    
    Parameters:
        A (numpy.ndarray): Matriks koefisien.
        b (numpy.ndarray): Vektor konstanta.
        
    Returns:
        numpy.ndarray: Solusi x dari sistem persamaan linear.
    """
    L, U = Lu_Gauss(A)
    n = len(A)
    y = np.zeros(n)
    x = np.zeros(n)
    
    
    for i in range(n):
        y[i] = b[i]
        for j in range(i):
            y[i] -= L[i][j] * y[j]
    
    
    for i in range(n-1, -1, -1):
        x[i] = y[i]
        for j in range(i+1, n):
            x[i] -= U[i][j] * x[j]
        x[i] /= U[i][i]
    
    return x

class TestLinearSystemSolver(unittest.TestCase):
    
    def test_solution(self):
        A = np.array([[3, 1, 4], [2, 1, 5], [6, 1, 2]])
        b = np.array([2, 4, 3])
        
        solusi = solusi_sistem_persamaan_linear(A, b)
        
        expected_solusi = np.array([5, -41, 7])
        
        self.assertTrue(np.allclose(solusi, expected_solusi))

if __name__ == '__main__':
    unittest.main()
