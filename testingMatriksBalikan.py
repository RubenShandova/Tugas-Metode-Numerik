import unittest
import numpy as np

def solusi_sistem_persamaan_linear(A, b):
    """
    Menyelesaikan sistem persamaan linear Ax = b menggunakan metode matriks balikan.
    
    Parameters:
        A (numpy.ndarray): Matriks koefisien.
        b (numpy.ndarray): Vektor konstanta.
        
    Returns:
        numpy.ndarray: Solusi x dari sistem persamaan linear.
    """
    A_inv = np.linalg.inv(A)
    x = np.dot(A_inv, b)
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

