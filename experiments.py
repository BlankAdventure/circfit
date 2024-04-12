# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 00:09:02 2023

@author: BlankAdventure
"""
import matplotlib.pyplot as plt
import itertools
import re
import circuit as ct
import numpy as np
from tabulate import tabulate
import pandas as pd

mapping = {'A': (lambda x: x.add_series_l(), lambda x: x),    #LS
           'B': (lambda x: x.add_series_c(), lambda x: -x),   #CS
           'C': (lambda x: x.add_parallel_l(), lambda x: x),  #LP
           'D': (lambda x: x.add_parallel_c(), lambda x: -x)  #CP
            }

all_components = ['A', 'B', 'C', 'D']
tline_components = ['A','D']

# Convert a list of results to a pandas dataframe
def to_pandas(results: list[tuple], include_model: bool=False):
    if include_model:
        subset = [ (x[0], x[0].circ, *x[1:]) for x in results]
        df = pd.DataFrame(subset)
        df.columns =['Model', 'Circuit', 'Min', 'Mean', 'p95', 'Max']
    else:
        subset = [ (x[0].circ, *x[1:]) for x in results]
        df = pd.DataFrame(subset)
        df.columns =['Circuit', 'Min', 'Mean', 'p95', 'Max']
    return df

def print_nice(results: list[tuple]) -> None:
    new = [ (x[0].circ, *x[1:]) for x in results]
    print(tabulate(new, headers=["Circuit","Min","Avg","p95","Max"],floatfmt=".2f",tablefmt="simple",showindex="always"))

def random_points_uniform (r_range: tuple[float, float], i_range: tuple[float, float], count: int, plot: bool = True) -> list[complex]:
    rr = np.random.uniform(low=r_range[0],high=r_range[1],size=count)
    ri = np.random.uniform(low=i_range[0],high=i_range[1],size=count)
    if plot:
        plt.plot(rr, ri, '.', alpha=0.5)
        plt.axis('equal')
        plt.grid()
        plt.show()
    return list(rr + 1j*ri)
    
def random_points_gaussian (znom: complex, r_std: float, i_std: float, corr: float, count: int, plot: bool = True) -> list[complex]:
    c0 = corr*r_std*i_std
    cov = np.asarray([[r_std**2, c0],[c0, i_std**2]])
    pts = np.random.multivariate_normal(mean=[np.real(znom),np.imag(znom)], cov=cov, size=count)
    
    # Remove any points with negative real value
    pts = np.asarray([p for p in pts if p[0] > 0])
    if plot:
        plt.plot(pts[:, 0], pts[:, 1], '.', alpha=0.5)
        plt.axis('equal')
        plt.grid()
        plt.show()
    return list(pts[:,0] + 1j*pts[:,1])

def single_run(zlist: list[complex], path: tuple, targ_func=ct.cost_max_swr) -> tuple:
    c = ct.Circuit()
    x0 = []
    for e in path:
        mapping[e][0](c)
        x0.append(  mapping[e][1](100)  )
    res = c.fit(zlist, x0, targ_func)
    swr = ct.swr_from_z(c.get_zin(zlist))
    p95 = np.percentile(swr, 95)
    return c, np.min(swr), np.mean(swr), p95, np.max(swr)

@ct.listify('depths')
def do_experiment(zlist, components, depths, targ_func):
    results = []
    for depth in depths:
        all_combs = get_combinations(components, depth)
        combs_clean,_ = clean(all_combs)
        for comb in combs_clean:
            res = single_run(zlist, comb, targ_func )
            results.append(res)
    return results

# Returns a list of tuples of all possible paths through the provided components
# do the specified depth.
def get_combinations(components: list[str], depth: int) -> list[tuple]:
    s = [components for _ in range(depth)]
    combs = list(itertools.product(*s))
    return combs



# Takes a list of component combinations and removes any paths containing
# the combinations listed in rejects. The listed pairs represent elements 
# pairings that can be reprsented by a single element instead of two: i.e.,
# a parallel cap followed by a parallel inductor
def clean(combinations: list[str]): #-> tuple[list(tuple), list(tuple)]:
    rejects = ['AB','BA','CD','DC']
    good_list = []
    bad_list = []
    for comb in combinations:
        as_str = ''.join(comb)
        if not re.findall(r'((\w)\2{1,})', as_str): #duplicates are not allowed
            if not any(c in as_str for c in rejects): #combinations in rejects are not allowed
                good_list.append(comb)
            else:
                bad_list.append(comb)
        else:
            bad_list.append(comb)            
    return good_list, bad_list


if __name__ == "__main__":
    #zlist = random_points_gaussian(30+1j*20, 10, 10, 0.4, 20)
    zlist = random_points_uniform((10,60),(5,20),5)
    
    res = do_experiment(zlist, all_components,2,ct.cost_max_swr)
    
    #res = do_experiment(30+1j*20, all_components,[2,3,4],ct.cost_max_swr)
    
    #print_nice(res)
    
    #res[3][0].draw()
    
    #znom = 30
    
    #zlist = random_
    
    #zlist = ct.get_test_points_guassian(znom,5,20)
    #res_nom = simple_fit(znom, zlist)
    
    #res_nom = do_experiment(list_all, [2], [znom], targ_func=ct.targ_func_multi_max_swr)
    #zlist = ct.get_test_points_guass(30, [[8,3.5],[2.5,4]], 500)
    #zlist = ct.get_test_points_guass(znom, [[8,7],[1,1]], 500)
    
    
    #res = do_experiment(list_all, [2,4], zlist)











