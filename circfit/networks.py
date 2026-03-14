# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 21:10:08 2026

@author: BlankAdventure
"""

import numpy as np
import networkx as nx
from functools import lru_cache
from typing import Protocol, overload, Any
from scipy.optimize import least_squares
from collections.abc import Iterable
import numpy.typing as npt
from collections.abc import Hashable

#matrix = np.ndarray[tuple[int, int],  np.dtype[np.complex64]]

#Matrix = npt.NDArray[np.complex64, np.complex64]
VectorComplex = npt.NDArray[np.complex64]
VectorFloat = npt.NDArray[np.float32]


#VectorComplex = np.ndarray[tuple[int], np.dtype[np.complex64]]
#VectorFloat = np.ndarray[tuple[int], np.dtype[np.float32]]


def rc_from_z(z_list: VectorComplex, z0:float=50) -> VectorComplex:
    return (z_list - z0) / (z_list + z0) 


def swr_from_rc (r_list: VectorFloat) -> VectorFloat:
    return (1+np.abs(r_list)) / (1-np.abs(r_list))

def swr_from_z (z_list: VectorComplex, z0:float=50) -> VectorFloat:
    return swr_from_rc(rc_from_z(z_list,z0))

def max_rc(z_list: VectorComplex) -> float:
    return np.max(np.abs(rc_from_z(z_list)))

def mean_rc(z_list: VectorComplex) -> float:
    return np.mean(np.abs(rc_from_z(z_list)))

def max_swr(z_list: VectorComplex) -> float:
    return np.max(swr_from_z(z_list))

def mean_swr(z_list: VectorComplex) -> float:
    return np.mean(swr_from_z(z_list))
    

def format_bounds(G: "Topo", bd: dict) -> tuple[list,list]:
    bounds = []
    x0 = []
    for u,v,k in G.edges:
        elem = G.edges[u,v,k]['type']
        bounds.append( bd[elem]["bounds"] )
        x0.append( bd[elem]["x0"] )
    bounds = [tuple( [ x[0] for x in bounds ]), tuple( [ x[1] for x in bounds ])]            
    return bounds, x0


def x_wrapper(params: VectorFloat, X: "XNetwork", z_list: VectorComplex) -> float:
    X.set_all_edges('weight', 1.0/np.conj(1.0j*params))
    zo = X.zin(z_list)
    return max_swr(np.asarray(zo))

def fit(G: "Topo", z_list: VectorComplex) -> tuple["XNetwork",Any]:
    
    if isinstance(z_list[0], complex):
        print (' **** reactance fitting ****')
        
        X = XNetwork(G)        
        bounds_dict = {"L": {'bounds': (0.0, np.inf),'x0': 10},
                       'C': {'bounds': (-np.inf, 0.0),'x0': -10},
                       }        
        bounds, x0 = format_bounds(G, bounds_dict)        
        
        func = lambda x: x_wrapper(x,X,z_list)        
        
        res = least_squares(func,x0,bounds=bounds,
                            loss='linear',
                            jac='3-point',
                            verbose=1,
                            method='trf',                            
                            x_scale=1)
        return X, res
        
        
    elif isinstance(z_list[0], tuple):
        # not implemented yet
        pass
 
    else:
        pass
    


class Base(nx.MultiGraph):
    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)    
        self.add_node("i")
        self.add_node("o")
        self.add_node("g")
        
    def get_all_edges(self, attrib: str) -> list[Any]:
        return list (nx.get_edge_attributes(self,attrib).values())
    
    def set_all_edges(self, attrib: str, values: list[str|float]):
        edges = list(self.edges)      
        edge_dict = dict(zip(edges, values))
        nx.set_edge_attributes(self, edge_dict, attrib)
    
    
class Topo(Base):    
    def add_element(self, n1: Hashable, n2: Hashable, elem: str):        
        self.add_edge(n1, n2, type=elem)

    
class XNetwork(Base):
    def add_element(self, n1: Hashable, n2: Hashable, Z:complex) -> int:
        return self.add_edge(n1, n2, weight=1.0/np.conj(Z)) 

    def _get_impedance(self, n1: Hashable, n2: Hashable) -> complex:
        L =  nx.laplacian_matrix(self).toarray()
        node_index = list(self.nodes)        
        N = L.shape[0]
        e = np.zeros(N, dtype=complex) 
        e[node_index.index(n1)] = 1.0
        e[node_index.index(n2)] = -1.0
        G_pinv = np.linalg.pinv(L)
        return e @ G_pinv @ e
       
    def _zin(self, zload: complex) -> complex:
       key = self.add_element("o","g",zload)
       z = self._get_impedance("i","g")
       self.remove_edge("o","g", key=key)
       return z

    def zin(self, z_list: VectorComplex) -> VectorComplex:
        arr = np.asarray(z_list)
        result = np.vectorize(self._zin)(arr)
        return result


class Circuit(Base):

    func_map = {"L": lambda f,x: 1j*2*np.pi*f*x,
                "C": lambda f,x: 1/(1j*2*np.pi*f*x),
                "R": lambda f,x: x
                }    

    def add_inductor(self, n1: Hashable, n2: Hashable, value: float):
        return self.add_edge(n1, n2, type="L", value=value)
        
    def add_capacitor(self, n1: Hashable, n2: Hashable, value: float):
        return self.add_edge(n1, n2, type="C", value=value)
        
    def add_resistor(self, n1: Hashable, n2: Hashable, value: float):
        return self.add_edge(n1, n2, type="R", value=value)
    
    @lru_cache
    def to_xnetwork(self, freq: float) -> XNetwork:
        X = XNetwork()
        for u, v, key in self.edges:
            x_r = Circuit.func_map[ self.edges[u,v,key]['type'] ](freq, self.edges[u,v,key]['value'])
            X.add_element(u, v, x_r)
        return X

    def _zin(self, freq: float, zload:complex) -> complex:
        return self.to_xnetwork(freq).zin(zload)

    def zin(self, z_list: VectorComplex) -> VectorComplex:
        temp = lambda row: self._zin(row[0],row[1])
        arr = np.asarray(z_list)
        result = np.apply_along_axis(temp, axis=0, arr=arr)
        return result
        

# def ranges2(z_list):
#     SC = 0.1
#     OC = 100000
    
#     f, _ = zip(*z_list)
    
#     f_min = min(f)
#     f_max = max(f)

#     L1 = SC/(2*np.pi*f_min)    
#     L2 = OC/(2*np.pi*f_min)    
#     L3 = SC/(2*np.pi*f_max)    
#     L4 = OC/(2*np.pi*f_max)    

#     C1 = 1/(SC*2*np.pi*f_min)    
#     C2 = 1/(OC*2*np.pi*f_min)    
#     C3 = 1/(SC*2*np.pi*f_max)    
#     C4 = 1/(OC*2*np.pi*f_max)    
    
#     Lrng = [L1, L2, L3, L4]
#     Crng = [C1, C2, C3, C4]
    
#     return {"L": (min(Lrng),max(Lrng)), "C": (min(Crng),max(Crng))}
    

# def ranges(z_list):
#     f, z = zip(*z_list)
    
#     z = np.asarray(z)
    
#     f_min = min(f)
#     f_max = max(f)
    
#     im = z.imag
#     xca = np.mean(im[im < 0]) if np.any(im < 0) else np.nan
#     xla = np.mean(im[im > 0]) if np.any(im > 0) else np.nan
    
    
    
#     print(f_min)
#     print(f_max)
    
#     print(xca)
#     print(xla)

#     L1 = xla / (2.0*np.pi*f_min)
#     L2 = xla / (2.0*np.pi*f_max)
    
#     C1 = 1 / (2.0*np.pi*f_min*xca)
#     C2 = 1 / (2.0*np.pi*f_max*xca)
    

#     print(L1)
#     print(L2)
#     print(C1)
#     print(C2)



c = Topo()
c.add_element("i",1,"L")
c.add_element(1,'g',"C")
c.add_element(1,'o',"L")


zl = [20-30j, 25-32j, 18-25j]

X,_ = fit(c, zl)
zo = X.zin( zl )

print(zo)
print(swr_from_z(zo))

# #%%

# zl = [ (1.1e8, 20-30j), (1.3e8, 25-32j), (1.2e8, 18-25j)]
# X,_ = fit(c, zl)

# zo = X.multi_zin(zl)
# swr_from_z(zo)

# #%%
# c = Circuit()
# c.add_inductor("i","o",56.27*1e-9)
# c.add_capacitor("o","g", 25.62*1e-12)
# #print(c.zin(1e8,60+1j*30))
# print(c.zin( (1e8,60+30j) ))



