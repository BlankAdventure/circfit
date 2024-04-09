# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 17:49:36 2023

@author: BlankAdventure

-class stores 'empty' circuit model
-once a fit is performed, element values get populated

-function to get input impedance given list of loads
---> Must specify element reactances -OR-
---> Must specify element values -OR-
---> Use fit values, if performed
---> Option to return drawing

cm.get_zin(zlist, reactances=[...], draw='t/f')
cm.get_zin(zlist, values=[...], draw='t/f')
cm.get_zin(zlist, draw='t/f') #if fit available

cm.populate_reactances()
cm.populate_circvals()

FITTING:
-Fit to given list of loads (1 or many)
---> Specify cost function 



cm.fit(z_list, cost_func)

cost_avg_swr
cost_max_swr
cost_deltaRI


"""
from scipy.optimize import least_squares
#from abc import ABC, abstractmethod, abstractproperty
#import matplotlib.pyplot as plt
#from functools import reduce, partial
import numpy as np
#import seaborn as sns
#from scipy.optimize import least_squares
import inspect


import schemdraw
import schemdraw.elements as elm

# Dictionary of valid unit strings and their multiple factor
units = {
         'p': 1e-12,
         'n': 1e-9,
         'u': 1e-6,
         'm': 1e-3,
         }

# Assorted helper functions
xL = lambda F, L: (1j*2*np.pi*F*L) 
xC = lambda F, C: (1/(1j*2*np.pi*F*C)) 
make_list = lambda x: [x] if type(x) is not list else x


# Decorator for ensuring single inputs are converted to a list
def listify(*argnames):
  def decorator(f):
      def decorated(*args, **kwargs):
          bound_args = inspect.signature(f).bind(*args, **kwargs)            
          for a in argnames:            
              bound_args.arguments[a] = make_list(bound_args.arguments[a])
          return f(*bound_args.args, **bound_args.kwargs)
      return decorated
  return decorator

# Decorator for replacing unit strings with numeric values
def unitsoff(*argnames):
    def decorator(f):
        def decorated(*args, **kwargs):
            bound_args = inspect.signature(f).bind(*args, **kwargs)            
            for a in argnames:            
                bound_args.arguments[a] = as_numeric(bound_args.arguments[a])
            return f(*bound_args.args, **bound_args.kwargs)
        return decorated
    return decorator


# def get_component_values(F, elements):
#     out = []
#     for e in elements:
#         if e > 0:
#             out.append( e / (2*np.pi*F) )
#         else:
#             out.append( -1/(2*np.pi*F*e) )
#     return out

# Converts a single unit string to a numeric string
def unitsToNumeric(str_val):
    unit = str_val[-1]
    try:
        scale = units[unit]
    except:
        print('Error! Invalid unit!')
    
    return float(str_val[:-1])*scale

# Given a list of values, identifies any unit strings and converts them to numeric values
#@listify('input_list')
def as_numeric(input_list):
    out = []
    for e in input_list:
        if isinstance(e, str):
            numeric = unitsToNumeric(e)
        else:
            numeric = e
        out.append(numeric)
    return out

@listify('zlist')
def rc(zlist, z0=50):
    return [(z - z0) / (z + z0) for z in zlist]

@listify('rlist')
def swr (rlist):
    return [(1+abs(r))/(1-abs(r)) for r in rlist]

def cost_max_swr(cm, x, zlist, zt=50):
    vals = cm.get_zin(loads=zlist, reactances_or_circvals=x)
    err = max(swr(rc(vals)))
    print(f'err: {err}')    
    return err

# def cost_avg_swr(cm, x, zlist, zt=50):
#     vals = []
#     for z in zlist:
#         out = cm.reactance_compute(x,z)
#         vals.append ( swr(rc(out)) )    
#     err = np.mean(vals)
#     print(f'out: {out:.2f} | err: {err}')    
#     return err

# def cost_max_absz(cm, x, zlist, zt=50):
#     vals = []
#     for z in zlist:
#         out = cm.reactance_compute(x,z)
#         vals.append ( out - zt )

#     err = max(abs(np.asarray(vals)))
#     print(f'out: {out:.2f} | err: {err}')    
#     return err

    
        
class Circuit():
    def __init__(self):
        self.func_list = []
        self.bounds = []
        self.circ = 'ZL'
        self.fit_values = None #reactances NOT component values!
        self.orientation = []

       
    def add_series(self):
        return lambda x, val: x+val
    
    def add_shunt(self):
        return lambda x, val: (x*val)/(x+val)

    def add_series_l(self):
        if not self.orientation or self.orientation[-1] == 'p':
            self.bounds.append( (0,np.inf)  )
            self.circ = ('SL-') + self.circ
            x_func = self.add_series()
            c_func = lambda x, F, val: x_func(x, xL(F, val))
            d_func = lambda label: elm.Inductor().right().label(label)
            self.func_list.append( (x_func, c_func, d_func) )
            self.orientation.append('s')
        else:
            print('Cannot add series after series!')            
 
    def add_parallel_l(self):
        if not self.orientation or self.orientation[-1] == 's':
            self.bounds.append( (0,np.inf)  )
            self.circ = ('PL-') + self.circ
            x_func = self.add_shunt()
            c_func = lambda x, F, val: x_func(x, xL(F, val))
            d_func = lambda label: elm.Inductor().down().label(label).hold()
            self.func_list.append( (x_func, c_func, d_func) )  
            self.orientation.append('p')
        else:
            print('Cannot add parallel after parallel!')

    def add_series_c(self):
        if not self.orientation or self.orientation[-1] == 'p':
            self.bounds.append( (-np.inf,0)  )
            self.circ = ('SC-') + self.circ
            x_func = self.add_series()
            c_func = lambda x, F, val: x_func(x, xC(F, val))
            d_func = lambda label: elm.Capacitor().right().label(label)
            self.func_list.append( (x_func, c_func, d_func) )
        else:
            print('Cannot add series after series!')

    def add_parallel_c(self):
        if not self.orientation or self.orientation[-1] == 's':
            self.bounds.append( (-np.inf,0)  )
            self.circ = ('PC-') + self.circ
            x_func = self.add_shunt()
            c_func = lambda x, F, val: x_func(x, xC(F, val))
            d_func = lambda label: elm.Capacitor().down().label(label).hold()
            self.func_list.append( (x_func, c_func, d_func) )     
            self.orientation.append('p')
        else:
            print('Cannot add parallel after parallel!')

    def draw(self, values=None, zin=None, zload=None, F=None):
        segs = str(c.orientation).count('s')   
        if self.orientation[0] == 'p' and self.orientation[-1] == 's':
            segs += 1
        if self.orientation[0] == 's' and self.orientation[-1] == 's':
            pass
        if self.orientation[0] == 's' and self.orientation[-1] == 'p':
            segs += 1
        if self.orientation[0] == 'p' and self.orientation[-1] == 'p':
            segs += 2
            
        with schemdraw.Drawing():       
            R = elm.ResistorIEC().down().hold()
            if self.orientation[0] == 'p': elm.Line().right()
            for e, func in enumerate(self.func_list):
                label = values[e] if values is not None else ""
                func[2](label)
            if self.orientation[-1] == 'p':
                elm.Line().right()
            ud = elm.Dot()
            elm.Line().right().at(R.end)
            for _ in range(segs-1):
                elm.Line().right()
            ld = elm.Dot()
            line = elm.Line().at(ld.center).to(ud.center).color('white')
            lstr = '$Z_{in}$'
            elm.ZLabel().at(line.center).left().label(lstr + f"={zin}", loc='top').color('black')

    def get_bounds(self):
        return [tuple( [ x[0] for x in self.bounds ]), tuple( [ x[1] for x in self.bounds ])]
    
    @listify('loads')
    def get_zin(self, loads, reactances_or_circvals=None, F=None):
        res = []      
        if reactances_or_circvals is not None and F is None:
            # Treat values as reactances
            res = [self.compute_with_reactances(z, reactances_or_circvals) for z in loads]
        elif reactances_or_circvals is not None and F is not None:
            # Treat values as circuit components
            res = [self.compute_with_circvals(z, reactances_or_circvals, F) for z in loads]
        elif reactances_or_circvals is None and F is None and self.fit_values is not None:
            # Use fit values, if available
            res = [self.compute_with_reactances(z, self.fit_values) for z in loads]
        elif reactances_or_circvals is None and F is None and self.fit_values is None:
            # Fit values not available
            print('Fit values not available')            
        else:
            print('Invalid arguments -> doing nothing')
        return res
            
    
    # Calculates single zin given element reactances (do not include j)
    @unitsoff('reactances')
    def compute_with_reactances(self, load, reactances):
        res = load
        for e, func in enumerate(self.func_list):
            res = func[0](res, 1j*reactances[e])
        return res
    
    # Calculates single zin given component values
    @unitsoff('component_values')
    def compute_with_circvals(self, load, component_values, F):
        res = load
        for e, func in enumerate(self.func_list):
            res = func[1](res, F, component_values[e])
        return res

    @listify('zlist')
    def fit(self, zlist, x0=None, cost_func=cost_max_swr):
        targ_wrapped = lambda x: cost_func(self, x, zlist, zt=50)
        res = least_squares(targ_wrapped,x0,bounds=self.get_bounds(),jac='2-point',verbose=1,method='trf',xtol=1e-5,x_scale=1)
        self.fit_values = res.x # These are reactances
        return res


zL = 30+1j*10 
   
c = Circuit()
c.add_series_l()
c.add_parallel_c()

print ( c.get_zin([zL, 20+1j*5],['2.307u', '2.599n' ],1e6) )
c.draw(['2.307u', '2.599n' ],zin=zL)

zlist = [30+1j*10, 20+1j*5, 35+1j*15]
res=c.fit(zlist,x0=[100,-100])

    
    
