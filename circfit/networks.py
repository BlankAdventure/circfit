# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 21:10:08 2026

@author: BlankAdventure
"""

import numpy as np
import networkx as nx
from functools import lru_cache
from typing import  Any, cast, TypeAlias
from scipy.optimize import least_squares, differential_evolution, basinhopping, dual_annealing
#from collections.abc import Iterable
import numpy.typing as npt
from collections.abc import Hashable


VectorComplex: TypeAlias = npt.NDArray[np.complex64]
VectorFloat: TypeAlias = npt.NDArray[np.float32]
ZList: TypeAlias = VectorComplex|complex|list[complex]

SC: float = 0.01
OC: float = 1e5


def rc_from_z(z_list: VectorComplex, z0:float=50) -> VectorComplex:
    '''get complex reflection coefficient from impedance'''
    return (z_list - z0) / (z_list + z0) 

def swr_from_rc (r_list: VectorComplex) -> VectorFloat:
    '''get SWR from complex reflection coefficient'''
    return (1+np.abs(r_list)) / (1-np.abs(r_list))

def swr_from_z (z_list: VectorComplex, z0:float=50) -> VectorFloat:
    '''get SWR from impedance'''
    return swr_from_rc(rc_from_z(z_list,z0))

def max_rc(z_list: VectorComplex) -> np.floating:
    '''
    determine the maximum absolute reflection coefficient value from a 
    list of impedances
    '''
    return np.max(np.abs(rc_from_z(z_list)))

def mean_rc(z_list: VectorComplex) -> np.floating:
    '''
    determine the average absolute reflection coefficient value from a 
    list of impedances
    '''
    return np.mean(np.abs(rc_from_z(z_list)))

def max_swr(z_list: VectorComplex) -> np.floating:
    '''
    determine the maximum SWR value from a list of impedances
    '''    
    return np.max(swr_from_z(z_list))

def mean_swr(z_list: VectorComplex) -> np.floating:
    '''
    determine the average SWR value from a list of impedances
    '''    
    return np.mean(swr_from_z(z_list))
    

def format_bounds(G: "Topo", bd: dict) -> tuple[list[float],list]:
    bounds = []
    x0 = []
    for u,v,k in G.edges:
        elem = G.edges[u,v,k]['type']
        bounds.append( bd[elem]["bounds"] )
        x0.append( bd[elem]["x0"] )
    return bounds, x0


def x_wrapper(params: VectorFloat, X: "XNetwork", z_list: VectorComplex) -> np.floating:
    X.set_all_edges('weight', 1.0/np.conj(-1.0j*params))
    zo = X.zin(z_list)
    return max_swr(zo)

def fit(G: "Topo", z_list: ZList, method: str = "diffevo") -> tuple["XNetwork",Any]:
    z_list = np.asarray(z_list)
    
    if isinstance(z_list[0], complex):
        print (' **** reactance fitting ****')        
        
        bounds_dict = {"l": {'bounds': (SC, OC),'x0': 20},
                       'c': {'bounds': (-OC, -SC),'x0': -10},
                       "x": {'bounds': (-OC, OC),'x0': np.random.uniform(-20,20)}
                       }        
        
        X = XNetwork(G)        
        bounds, x0 = format_bounds(G, bounds_dict)        
        func = lambda x: x_wrapper(x,X,z_list)

        if method == "basin":        
            minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bounds}
            res = basinhopping(func, x0, minimizer_kwargs=minimizer_kwargs, disp=False)        
        elif method == "diffevo":        
            res = differential_evolution(func, bounds)
        elif method == "anneal":
            res = dual_annealing(func, bounds)
        elif method == "lstsqrs":
            bounds = [tuple( [ x[0] for x in bounds ]), tuple( [ x[1] for x in bounds ])]
            res = least_squares(func,x0,bounds=bounds,                            
                             jac='3-point',
                             verbose=0,
                             method='trf'
                             )
        else:
            print('invalid method')
                            
        return X, res
        
        
    elif isinstance(z_list[0], tuple):
        print (' **** circuit fitting ****')
        # not implemented yet       
 
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
        
    def is_valid(self) -> bool:
        Q = self.copy()
        Q.remove_node("g")
        return nx.has_path(Q,"i","o")
    
class Topo(Base):    
    def add_element(self, n1: Hashable, n2: Hashable, elem: str):   
        if isinstance(elem, str) and len(elem) == 1 and elem.lower() in 'lcx':        
            self.add_edge(n1, n2, type=elem.lower())
        else:
            raise ValueError ("Element must be L, C, or X.")
    
    def using(self):
        pass
    
    def fit(self):
        pass
    
class XNetwork(Base):
    def add_element(self, n1: Hashable, n2: Hashable, Z:complex) -> int:
        if abs(Z) > 0:
            return self.add_edge(n1, n2, weight=1.0/Z)
        else:
            raise ValueError ("|Z| must be > 0")
            
    def _get_impedance(self, n1: Hashable, n2: Hashable) -> complex:
        L =  nx.laplacian_matrix(self).toarray()
        node_index = list(self.nodes)        
        N = L.shape[0]
        e = np.zeros(N, dtype=complex) 
        e[node_index.index(n1)] = 1.0
        e[node_index.index(n2)] = -1.0
        G_pinv = np.linalg.pinv(L)
        return cast(complex, e @ G_pinv @ e)
       
    def _zin(self, z_load: complex) -> complex:
       key = self.add_element("o","g",z_load)
       z = self._get_impedance("i","g")
       self.remove_edge("o","g", key=key)
       return z

    def zin(self, z_list: ZList) -> VectorComplex:
        arr = np.asarray(z_list)
        result = np.vectorize(self._zin)(arr)
        return result
    
    def __str__(self) -> str:   
        result = ''        
        for u, v, key in self.edges(data=True):
            result += f'{u}-{v}: z = {1.0/key["weight"]:.2f}\n'
        return result



class Circuit(Base):

    func_map = {"L": lambda f,x: 1j*2*np.pi*f*x,
                "C": lambda f,x: 1/(1j*2*np.pi*f*x),
                "R": lambda f,x: x
                }    

    def add_inductor(self, n1: Hashable, n2: Hashable, value: float):
        if value > 0:
            self.add_edge(n1, n2, type="L", value=value)
        else:
            raise ValueError ("Inductance must be > 0.")

    def add_capacitor(self, n1: Hashable, n2: Hashable, value: float):
        if value > 0:        
            self.add_edge(n1, n2, type="C", value=value)
        else:
            raise ValueError ("Capacitance must be > 0.")
        
    def add_resistor(self, n1: Hashable, n2: Hashable, value: float):
        if value > 0:
            self.add_edge(n1, n2, type="R", value=value)
        else:
            raise ValueError("Resistance must be > 0.")
    
    
    @lru_cache
    def to_xnetwork(self, freq: float) -> XNetwork:
        X = XNetwork()
        for u, v, key in self.edges:
            x_r = Circuit.func_map[ self.edges[u,v,key]['type'] ](freq, self.edges[u,v,key]['value'])
            X.add_element(u, v, x_r)
        return X

    def _zin(self, freq: float, z_load: VectorComplex) -> VectorComplex:
        return self.to_xnetwork(freq).zin(z_load)

    def zin(self, z_list: ZList) -> VectorComplex:        
        arr = np.asarray(z_list)
        result = np.apply_along_axis(lambda row: self._zin(row[0],row[1]), axis=0, arr=arr)
        return result
        

    






#%%
c = Topo()
#c.add_element("i",1,"C")
#c.add_element(1,'g',"L")
#c.add_element(1,'o',"C")

#c.add_element("i",1,"L")
#c.add_element(1,'g',"C")
#c.add_element(1,'o',"L")

#c.add_element("i","o","X")
#c.add_element("o",'g',"X")

c.add_element("i","o","c")
c.add_element("o",'g',"l")

zl = [20-30j] #, 25-32j, 18-25j]

X,res = fit(c, zl,"diffevo")
zo = X.zin( zl )
print(X)
print(zo)
print(swr_from_z(zo))

#c.using('diffevo').fit(zl)

# zl = [ (1.1e8, 20-30j), (1.3e8, 25-32j), (1.2e8, 18-25j)]
# X,_ = fit(c, zl)

# zo = X.multi_zin(zl)
# swr_from_z(zo)

#%%
c = Circuit()
c.add_inductor("i","o",56.27*1e-9)
c.add_capacitor("o","g", 25.62*1e-12)
#print(c.zin(1e8,60+1j*30))
print(c.zin( (1e8,60+30j) ))

x = XNetwork()
x.add_element("i","o",-10j)
print( x.zin(15+20j) )





