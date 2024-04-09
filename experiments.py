# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 00:09:02 2023

@author: BlankAdventure
"""

import itertools
import re
import circ_test002 as ct
import numpy as np

mapping = {'A': (lambda x: x.add_series_l(), lambda x: x),    #LS
           'B': (lambda x: x.add_series_c(), lambda x: -x),   #CS
           'C': (lambda x: x.add_parallel_l(), lambda x: x),  #LP
           'D': (lambda x: x.add_parallel_c(), lambda x: -x)  #CP
            }

list_all = ['A', 'B', 'C', 'D']
list_tline = ['A','D']

def simple_fit(zl,zlist):
    valid = get_combs(list_all, 2)
    results = []
    for path in valid:
        c = ct.Circuit()
        x0 = []
        for e in path:
            mapping[e][0](c)
            x0.append(  mapping[e][1](100)  )
        res = c.do_single_fit(x0, zl)
        swr = vswr(c, zlist)
        p95 = np.percentile(swr, 95)
        results.append(  (c.circ,f'{np.mean(swr):.2f}',f'{np.max(swr):.2f}',f'{p95:.2f}')  )
    return results

def single_run(valid, zlist, targ_func=ct.targ_func_multi_avg_swr):
    results = []
    for path in valid:
        c = ct.Circuit()
        x0 = []
        for e in path:
            mapping[e][0](c)
            x0.append(  mapping[e][1](100)  )
        res = c.do_multi_fit(x0, zlist, targ_func)
        swr = vswr(c, zlist)
        p95 = np.percentile(swr, 95)
        results.append(  (c.circ,f'{np.mean(swr):.2f}',f'{np.max(swr):.2f}',f'{p95:.2f}')  )
    return results

def do_experiment(opt_list, depths, zlist, targ_func=ct.targ_func_multi_avg_swr):
    results = []
    for depth in depths:
        valid = get_combs(opt_list, depth)
        res = single_run(valid,zlist,targ_func=targ_func)
        results.append(res)
    return results

def get_combs(options_list, depth):        
    s = [options_list for _ in range(depth)]
    combs = list(itertools.product(*s))
    valid,_ = clean(combs)
    return valid

def vswr (cm, zlist):
    zo = cm.fit_compute(1e6, zlist)        
    refco = [ct.rc(z) for z in zo]
    vswr = [ct.swr(x) for x in refco]
    return vswr

def clean(list_in):
    rejects = ['AB','BA','CD','DC']
    good_list = []
    bad_list = []
    for comb in list_in:
        as_str = ''.join(comb)
        if not re.findall(r'((\w)\2{1,})', as_str):
            if not any(c in as_str for c in rejects):
                good_list.append(comb)
            else:
                bad_list.append(comb)
        else:
            bad_list.append(comb)
            
    return good_list, bad_list



znom = 30

zlist = ct.get_test_points_guass(znom, [[8,7],[4,1]], 50)
res_nom = simple_fit(znom, zlist)

#res_nom = do_experiment(list_all, [2], [znom], targ_func=ct.targ_func_multi_max_swr)


#zlist = ct.get_test_points_guass(30, [[8,3.5],[2.5,4]], 500)

#zlist = ct.get_test_points_guass(znom, [[8,7],[1,1]], 500)

res = do_experiment(list_all, [2,4], zlist)











