# Lista 7 - Prog 2 - Jader Duarte
# Questão 1
def find_judge(n,trust):
    pessoas={p+1 for p in range(n)}
    for i in trust:
        pessoas.discard(i[0]) # como todo mundo confiam no juiz, todos os não juizes confiam em alguem. Portanto o unico que não confia em ninguem é o juiz.  
    if pessoas:
        return pessoas.pop()
    else:
        return -1
    
# Questão 4
# Parte no git 
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
    
class Domain:
    min = None
    max = None

    def __contains__(self, x):
        raise NotImplementedError
    
    def __repr__(self):
        raise NotImplementedError

    def __str__(self):
        return self.__repr__()
    
    def copy(self):
        raise NotImplementedError 


class Interval(Domain):
    def __init__(self, p1, p2):
        self.inff, self.supp = min(p1, p2), max(p1, p2)
    
    @property
    def min(self):
        return self.inff

    @property
    def max(self):
        return self.supp
    
    @property
    def size(self):
        return (self.max - self.min)
    
    @property
    def haf(self):
        return (self.max + self.min)/2.0
    
    def __contains__(self, x):
        return  np.all(np.logical_and(self.inff <= x, x <= self.supp))

    def __str__(self):
        return f'[{self.inff:2.4f}, {self.supp:2.4f}]' 

    def __repr__(self):
        return f'[{self.inf!r:2.4f}, {self.supp!r:2.4f}]'
    
    def copy(self):
        return Interval(self.inff, self.supp)


class RealFunction:
    f = None
    prime = None
    domain = None
    
    def eval_safe(self, x):
        if self.domain is None or x in self.domain:
            return self.f(x)
        else:
            raise Exception("The number is out of the domain")

    def prime_safe(self, x):
        if self.domain is None or x in self.domain:
            return self.prime(x)
        else:
            raise Exception("The number is out of the domain")
        
    def __call__(self, x) -> float:
        return self.eval_safe(x)
    
    def plot(self):
        fig, ax = plt.subplots()
        X = np.linspace(self.domain.min, self.domain.max, 100)
        Y = self(X)
        ax.plot(X,Y)
        return fig, ax



def bissect(f: RealFunction, 
            search_space: Interval, 
            erroTol: float = 1e-4, 
            maxItr: int = 1e4, 
            eps: float = 1e-6 ) -> Interval:
    count = 0
    ss = search_space.copy()
    err = ss.size/2.0
    fa, fb = f(ss.min), f(ss.max)
    if fa * fb > -eps:
        if abs(fa) < eps:
            return Interval(ss.min, ss.min)
        elif abs(fb) < eps:
            return Interval(ss.max, ss.max)
        else:
            raise Exception("The interval extremes share the same signal;\n employ the grid search method to locate a valid interval.")
    while count <= maxItr and err > erroTol:
        count += 1
        a, b, m =  ss.min, ss.max, ss.haf
        fa, fb, fm = f(a), f(b), f(m)
        if abs(fm) < eps:
            return Interval(m, m)
        elif fa * fm < -eps:
            ss = Interval(a, m)
        elif fb * fm < -eps:
            ss = Interval(m, b)
    return ss


def grid_search(f: RealFunction, domain: Interval = None, grid_freq = 8) -> Interval:
    if domain is not None:
        D = domain.copy()
    else:
        D = f.domain.copy()
    L1 = np.linspace(D.min, D.max, grid_freq)
    FL1 = f(L1)
    TI = FL1[:-1]*FL1[1:]
    VI = TI <= 0
    if not np.any(VI):
        return None
    idx = np.argmax(VI)
    return Interval(L1[idx], L1[idx+1])

# Questão 5


class interpolater:

    def evaluate(self, X):
        raise NotImplementedError

    def __call__(self,  X):
        return self.evaluate(X)


class VandermondeMatrix(interpolater):
    def __init__(self, x, y):
        if len(x) != len(y):
            raise RuntimeError(f"Dimensions must be equal len(x) = {len(x)} != len(y) = {len(y)}")
        self.data = [x, y]
        self._degree = len(x) -1
        self._buildMatrix()
        self._poly = np.linalg.solve(self.matrix, self.data[1])

    def _buildMatrix(self):
        self.matrix = np.ones([self._degree+1, self._degree+1])
        for i, x in enumerate(self.data[0]):
            self.matrix[i, 1:] = np.multiply.accumulate(np.repeat(x, self._degree))
    
    def evaluate(self, X):
        r = 0.0
        for c in self._poly[::-1]:
            r = c+r*X
        return r
## Resposta 
class lagrangepol(interpolater):
  def __init__(self,x,y):
    self.pol= lagrange(x,y)
  
  def evaluate(self, X):
        return self.pol(X)
###
def random_sample(intv, N):
    r = np.random.uniform(intv[0], intv[1], N-2)
    r.sort()
    return np.array([intv[0]] + list(r) + [intv[1]])

def error_pol(f, P, intv, n = 1000):
    x = random_sample(intv, n)
    vectError = np.abs(f(x)-P(x))
    return np.sum(vectError)/n, np.max(vectError)

if __name__=="__main__":
    # Q1
    t = [ [ 1 , 2 ] , [ 1 , 3 ] , [ 2 , 3 ] ]
    n = 3
    print(f"O juiz é o {find_judge(n, t)}")  # Saída: 3
    # Q4
    d = Interval(-1.0, 2.0)
    print(d)

    nt = np.linspace(d.min-.1, d.max+1, 5)

    for n in nt:
        sts = 'IN' if n in d else 'OUT'
        print(f'{n} is {sts} of {d}')

    class funcTest(RealFunction):
        f = lambda self, x : np.power(x, 2) - 1
        prime = lambda self, x : 2*x
        domain = Interval(-2, 2)

    ft = funcTest()
    ND = grid_search( ft, grid_freq=12)
    print(bissect(ft, search_space=ND))
    from time import time 

    DataX = [10.7       , 11.075     , 11.45      , 11.825     , 12.2       , 12.5]
    DataY = [-0.25991903,  0.04625002,  0.16592075,  0.13048074,  0.13902777, 0.2]

    Pvm = VandermondeMatrix(DataX, DataY)

    X = np.linspace(min(DataX)-0.2, max(DataX)+0.2, 100)
    Y_vm = Pvm(X)

    _, ax = plt.subplots(1)
    ax.plot(X,Y_vm)
    ax.axis('equal')
    ax.plot(DataX, DataY, 'o')
    plt.show()
    ## Exemplo lagrange 
    Lagrange_testezinho= lagrangepol(DataX,DataY)
    #print(Lagrange_testezinho(10.7))
    Y_lagrange=Lagrange_testezinho(X)

    _, ax = plt.subplots(1)
    ax.plot(X,Y_lagrange)
    ax.axis('equal')
    ax.plot(DataX, DataY, 'o')
    plt.show()

    ##comparando velocidade
    t_0=time()
    VandermondeMatrix(DataX, DataY)
    t_vm= time()-t_0
    start_time=time()
    lagrangepol(DataX, DataY)
    t_lag=time()-start_time
    print(f"""
    tempo de lagrange:{t_lag}
    tempo da matriz de fulaninho: {t_vm}""")


    ### Questão 3
    ### Usando os codigos das questões anteriores
    Altura=[200,400,600,800,1000,1200,1400]
    Temperatura=[15,9,5,3,-2,-5,15]
    aproximação_3=lagrangepol(Altura,Temperatura)
    r_a=
    r_b=
    print( f"""
Questão 3, resposta
    Utilizando pôlinomio de lagrange para aproximar uma função que relacione a altura e a temperatura com os dados que temos:
    a) 
    b) Utilizando pôlinomio de lagrange para aproximar uma função que relacione a altura e a temperatura com os dados que temos, vemos que quando o avião estava a 700 metros de altura sua temperatura era de {aproximação_3(700)} °C""")
