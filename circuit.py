# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 17:49:36 2023

@author: BlankAdventure


"""
import math
from collections.abc import Iterable
from typing import Union
from scipy.optimize import least_squares
import numpy as np
import inspect
import re



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
make_list = lambda x: [x] if not isinstance(x, Iterable) else x


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
                bound_args.arguments[a] = drop_units(bound_args.arguments[a])
            return f(*bound_args.args, **bound_args.kwargs)
        return decorated
    return decorator




@listify('input_list')
def add_units(input_list: list[float]) -> list[str]:
    """
    Converts list of numeric values to list of unit strings    

    Parameters
    ----------
    input_list : list[float]
        Numeric values to be converted to unit strings

    Returns
    -------
    list[str]
        Converted list of unit strings

    """
    out = []
    for e in input_list:
        exponent = math.floor(math.log10(abs(e)))
        unit = min(units.keys(), key=lambda x: abs(exponent - math.log10(abs(units[x]))))
        new_val = e / units[unit]
        out.append(f'{new_val:.2f}{unit}')
    return out


# Given a list of values, identify and convert unit strings to numeric values
@listify('input_list')
def drop_units(input_list: list[Union[str, float]]) -> list[float]:
    """
    Given a list of values, identify and convert unit strings to numeric values    

    Parameters
    ----------
    input_list : list[Union[str, float]]
        Values or unit strings to convert to numeral

    Raises
    ------
    ValueError
        Invalid unit supplied

    Returns
    -------
    list[float]
        List of converted values (i.e, numeral)

    """
    out = []
    for e in input_list:
        if isinstance(e, str):
            try:
                numeric = float(e[:-1]) * units[e[-1]]
            except KeyError:
                raise ValueError(f'Unrecognized unit: {e[-1]}')
        else:
            numeric = e
        out.append(numeric)
    return out

@listify('zlist')
def rc_from_z(zlist, z0=50):
    return [(z - z0) / (z + z0) for z in zlist]

@listify('rlist')
def swr_from_rc (rlist):
    return [(1+abs(r))/(1-abs(r)) for r in rlist]

@listify('zlist')
def swr_from_z (zlist, z0=50):
    return swr_from_rc(rc_from_z(zlist,z0))

def cost_max_swr(cm, x, zlist, zt=50):
    vals = cm.get_zin(loads=zlist, reactances_or_circvals=x)
    err = max(swr_from_z(vals))
    return err

def cost_mean_swr(cm, x, zlist, zt=50):
    vals = cm.get_zin(loads=zlist, reactances_or_circvals=x)
    err = np.mean(swr_from_z(vals))
    return err

def cost_max_absz(cm, x, zlist, zt=50):
    vals = cm.get_zin(loads=zlist, reactances_or_circvals=x)
    err = max(np.abs(np.asarray(zlist) - np.asarray(vals)))
    return err


@listify('elements')
def get_component_values(F: float, elements: list[float]) -> list:
    out = []
    for e in elements:
        if e > 0:
            out.append( e / (2*np.pi*F) )
        else:
            out.append( -1/(2*np.pi*F*e) )
    return out
    
        
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
            self.orientation.append('s')
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

    def draw(self, values=None, zin=None, zload=None, F=None, for_web=False ):
        segs = str(self.orientation).count('s')   
        if self.orientation[0] == 'p' and self.orientation[-1] == 's':
            segs += 1
        if self.orientation[0] == 's' and self.orientation[-1] == 's':
            pass
        if self.orientation[0] == 's' and self.orientation[-1] == 'p':
            segs += 1
        if self.orientation[0] == 'p' and self.orientation[-1] == 'p':
            segs += 2
        
        
        if values is None and F is None and self.fit_values is not None:
            values =  [f'{x:.1f}i' for x in self.fit_values]
        elif values is None and F is not None and self.fit_values is not None:
            values = add_units( get_component_values(F, self.fit_values) )
            comp_str = re.sub('[PSZ-]', '', self.circ[:-1]).replace('C','F').replace('L','H')            
            values = [i + j for i, j in zip(values, comp_str)]
        elif values is not None and F is not None:
            values = add_units( get_component_values(F, values) )
            comp_str = re.sub('[PSZ-]', '', self.circ[:-1]).replace('C','F').replace('L','H')            
            values = [i + j for i, j in zip(values, comp_str)]            
        else:
            pass       
        
        with schemdraw.Drawing(show = not for_web, canvas='svg') as d:            
            load_label = '$Z_{load}$'
            if zload: load_label +=  f'={zload}'
            R = elm.ResistorIEC().down().hold().label(load_label)            
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
            in_label = '$Z_{in}$'
            if zload: in_label +=  f'={zin}'
            elm.ZLabel().at(line.center).left().label(in_label, loc='top').color('black')
        return d.get_imagedata('svg')

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
        res = least_squares(targ_wrapped,x0,bounds=self.get_bounds(),jac='2-point',verbose=0,method='trf',xtol=1e-5,x_scale=1)
        self.fit_values = res.x # These are reactances
        return res

if __name__ == "__main__":
    zL = 30+1j*10 
       
    c = Circuit()
    #c.add_series_l()
    #c.add_parallel_c()
    
    c.add_parallel_l()
    c.add_series_c()
    c.add_parallel_l()
    
    #rint ( c.get_zin([zL, 20+1j*5],['2.307u', '2.599n' ],1e6) )
    #c.draw(['2.307u', '2.599n' ],zin=zL)
    
    zlist = [30+1j*10, 20+1j*5, 35+1j*15]
    zavg = np.mean(zlist)
    
    res1 = c.fit(zlist,x0=[100,-100],cost_func=cost_mean_swr)
    
    s1 = swr_from_z(c.get_zin(zlist))
    
    res2 = c.fit(zavg,x0=[100,-100],cost_func=cost_mean_swr)
    
    s2 = swr_from_z(c.get_zin(zlist))

    
    
