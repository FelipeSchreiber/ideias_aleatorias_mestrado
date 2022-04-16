import numpy as np

"""
Python implementation of "AN O(n) ALGORITHM FOR GENERATING UNIFORM RANDOM
VECTORS IN n-DIMENSIONAL CONES"
"""

def generate_angle(theta_0,n):
    min_ = min(theta_0,np.pi/2)
    h = (n-2)*np.log(np.sin(min_))
    while True:
        u = np.random.uniform(low=0.0,high=1.0)
        theta = np.random.uniform(low=0.0,high=theta_0)
        f = h + np.log(u)
        if(f < (n-2)*np.log(np.sin(theta))):
            return theta

def generate_point_on_sphere(n):
    x = np.zeros(n)
    for i in range(n):
        x[i] = np.random.normal(loc=0,scale=1)
    x /= np.linalg.norm(x)
    return x

def rotate_vector_from_nth_canonical_basis(x,mu,n):
    P = np.zeros((n,2))
    P[-1][0] = 1
    for i in range(n-1):
        P[i][1] = mu[i]/np.sqrt(1-mu[-1]**2)
    P[-1][1] = 0
    G = np.array([
                    [mu[-1], -np.sqrt(1-mu[-1]**2)],
                    [np.sqrt(1-mu[-1]**2), mu[-1] ]
                ])
    I = np.eye(2)
    y = x + P@(G-I)@P.T@x
    return y

def generate_point(mu,theta_0,n):
    theta = generate_angle(theta_0,n)
    x = np.zeros(n)
    x[:-1] = generate_point_on_sphere(n-1)
    x *= np.sin(theta)
    x[-1] = np.cos(theta)
    if 1 != mu[-1]:
    	x = rotate_vector_from_nth_canonical_basis(x,mu,n)
    return x
   
#def compute_rotation_matrix(v,theta,n):
	 
"""
Implementation of "General n-Dimensional Rotations"
V is the rotation basis, alpha is the openess of the cone 1, beta the openess of cone 2 and phi the openess of intersection. N is the number of dimensions
"""

"""
def rotate_along_axis(v,alpha,beta,phi,n):
	angle = np.pi/2 -alpha -beta + 2*phi
"""	
	
