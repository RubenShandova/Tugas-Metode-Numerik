import numpy as np

def dekomposisi_crout(A):
    """
    Melakukan dekomposisi Crout pada matriks A.
    
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
            sum = 0
            for k in range(i):
                sum += L[i][k] * U[k][j]
            U[i][j] = A[i][j] - sum
        
        for j in range(i+1, n):
            sum = 0
            for k in range(i):
                sum += L[j][k] * U[k][i]
            L[j][i] = (A[j][i] - sum) / U[i][i]
    
    return L, U

def solusi_sistem_persamaan_linear(A, b):
    """
    Menyelesaikan sistem persamaan linear Ax = b menggunakan metode dekomposisi Crout.
    
    Parameters:
        A (numpy.ndarray): Matriks koefisien.
        b (numpy.ndarray): Vektor konstanta.
        
    Returns:
        numpy.ndarray: Solusi x dari sistem persamaan linear.
    """
    L, U = dekomposisi_crout(A)
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

A = np.array([[3, 1, 4], [2, 1, 5], [6, 1, 2]])
b = np.array([[2], [4], [3]])

solusi = solusi_sistem_persamaan_linear(A, b)
print("Solusi x dari sistem persamaan linear adalah:", solusi)
