import numpy as np
from scipy.integrate import solve_bvp
import sympy as sp
from sympy import diff, simplify,Matrix
import math

def christoffel_symbols_diagonal_metric(g,u,v):
    assert sp.shape(g) == (2,2)
    r1 = [-1/(2*g[0,0]) * diff(g[0,0],u), -1/(g[0,0]) * diff(g[0,0],v), 1/(2*g[0,0]) * diff(g[1,1],u)]

    r2 = [1/(2*g[1,1]) * diff(g[0,0],v), -1/(g[1,1]) * diff(g[1,1], u), -1/(2 * g[1,1]) * diff(g[1,1], v)]
    return [[simplify(x) for x in r1],[simplify(x) for x in r2]]

def geodesic_eq(x,y,k=None):
    # x : m,1 mesh size
    # y : 4,m
    du0 = y[1]
    dv0 = y[3]

    u0 = y[0]
    v0 = y[2]

    u,v = sp.symbols('u v')
    points = [[c.subs({u:u0[0], v:v0[0]}) for c in r] for r in k]
    # 111 112 122
    # 211 212 222

    derivs = np.vstack([du0,
              points[0][2]*dv0*dv0 + points[0][1]*dv0*du0 + points[0][0]*du0*du0,
              dv0,
              points[1][0]*du0*du0 + points[1][1]*du0*dv0 + points[1][2]*dv0*dv0])
    return derivs

def bc(ya,yb, p=None):
    u0,v0,u1,v1 = p
    return np.array([ya[0]-u0,yb[0]-v0,ya[2]-u1,yb[2]-v1])

def solve_geodesic(points,step,christoffel_symbols):
    s = np.arange(0, 1, step)
    guess = np.zeros((4,len(s)))
    return solve_bvp(lambda x,y: geodesic_eq(x,y,christoffel_symbols), lambda ya,yb: bc(ya,yb,points), s, guess,verbose=1)

def sphere(u,v):
    return np.sin(v) * np.cos(u),np.sin(v) * np.sin(u),np.cos(v)
    #return np.cos(v) * np.cos(u), np.cos(v) * np.sin(u), np.sin(v)

def plot_sphere(christoffel_symbols):
    import matplotlib.pylab as plt

    step = 0.015
    points = [1, 1, math.pi / 2, math.pi / 2]
    soln = solve_geodesic(points, step, christoffel_symbols)
    geodesic = soln.y
    N = geodesic.shape[1]
    u, v = plt.meshgrid(np.linspace(0, 2*np.pi, N), np.linspace(0, 2*np.pi, N))
    x,y,z = sphere(u,v)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    import matplotlib.cm as cm
    theCM = cm.get_cmap()
    theCM._init()
    alphas = -.5 * np.ones(theCM.N)
    theCM._lut[:-3, -1] = alphas
    ax.plot_surface(x, y, z,cmap=theCM)

    u,v = [points[0],points[2]],[points[1],points[3]]
    x,y,z = sphere(u,v)
    ax.scatter(x,y,z,c="r")

    # plot the parametrized data on to the sphere
    u, v = geodesic[0, :], geodesic[2,:]
    x,y,z = sphere(u,v)

    ax.plot(x, y, z,"r")
    plt.show()

if __name__ == "__main__":
    u,v = sp.symbols('u v')



    #g = Matrix([[sp.cos(u)**2,0.0],\
    #               [0.0,1]])

    g = Matrix([[1.0,0.0],\
                   [0.0,sp.sin(u)**2]])
    christoffel_symbols = christoffel_symbols_diagonal_metric(g,u,v)
    expected = [[0,-2*sp.tan(v),0],[1/2 * sp.sin(2*v), 0, 0]]
    print(christoffel_symbols)
    #assert sp.Matrix(christoffel_symbols) == expected
    plot_sphere(christoffel_symbols)




