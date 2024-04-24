# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 17:38:28 2024

@author: BlankAdventure
"""
import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

import asyncio
import circuit as ct
import experiments as ex
import plotly.graph_objects as go
from nicegui import ui
import numpy as np
from functools import wraps, partial


# min/max settings for sliders/impedance values
max_r = 200
min_r = 0
max_i = 100
min_i = -100
points = 5

section_style = ['text-white','font-bold','bg-gray-500','w-full']                                                           
                                                         

# aggrid table options. Must match column names in results df
table_options = {'columnDefs': [{'headerName': 'Circuit', 'field': 'Circuit','sortable': False, 'width': 240},
                       {'headerName': 'Min', 'field': 'Min','sortable': True},
                       {'headerName': 'Mean', 'field': 'Mean','sortable': True},
                       {'headerName': 'p95', 'field': 'p95','sortable': True},
                       {'headerName': 'Max', 'field': 'Max','sortable': True}],
        'rowHeight': 25,
        'rowSelection': 'single',
        'domLayout': 'normal',
        'autoSizeStrategy': {'type': 'fitGridWidth'},
        }


smith_dict = {'imag':[],
              'real':[],
              'opacity': 0.7,
              'marker_symbol':'circle',
              'marker_size':8,
              'marker_color':'red',
              'mode':'markers',
              'showlegend':False,
              'name':'inputs'}


# Decorator to convert standard funcs to async (func must be called *from* async)
def wrap(func):
    @wraps(func)
    async def run(*args, loop=None, executor=None, **kwargs):
        if loop is None:
            loop = asyncio.get_event_loop()
        pfunc = partial(func, *args, **kwargs)
        return await loop.run_in_executor(executor, pfunc)
    return run

# Update fit results table and circuit image 
def refresh_table(app, res_df) -> None:
    def update_outputs(event) -> None:
        idx = int(event.args['rowId'])
        cm = res_df.loc[idx, 'Model']
        zin = cm.get_zin(app.zlist)
        rc = [z/50 for z in zin] #Normalize to 50 ohms
        app.fig.update_traces(patch={"imag": np.imag(rc), "real": np.real(rc)}, selector = {'name':'outputs'})
        app.plot.update()    
        
        app.imageDiv.clear()
        with app.imageDiv:            
            freq = ui.input(value=None,placeholder='<none>',label='Freq [Hz]').classes('w-28 bg-white').props(
                'clearable square outlined dense input-class="font-mono" stack-label').style(
                    'position: absolute; z-index: 1; top: -8px; right: 0px;')                
            h = ui.html().classes('max-w-max').style('margin: 0 auto; justify-content: center;') 
            h.set_content(cm.draw(for_web=True).decode('utf-8'))
            freq.on('blur', lambda: h.set_content(cm.draw(for_web=True,F=float(freq.value) if freq.value else None).decode('utf-8')))
                        
    if res_df is not None:
        subset = res_df.loc[:, res_df.columns != 'Model'] #We only want the numerical results data, not the objects column
        x = len(res_df)
        if x > 18:
            style =  f'height: {(18+1)*25 + 15}px;'
        else:
            style =  f'height: {(x+1)*25 + 15}px;'

        app.gridDiv.clear()
        with app.gridDiv:
            ui.aggrid.from_pandas(subset, 
                        options=table_options).on('rowDoubleClicked', 
                        update_outputs, ['rowId'] ).classes().style(style)
                                                  
     
# Do the fit!
@wrap
def fit(app) -> None:
    if app.zlist:
        app.overlay.set_visibility(True)
        depth = [int(i.strip()) for i in app.levels.value.split(',')]
        result = ex.do_experiment(app.zlist, ex.all_components,depth,ct.cost_max_swr)
        refresh_table( app, ex.to_pandas(result, include_model=True).round(2) )
        app.overlay.set_visibility(False)


    
# Generates a new list of random load impedances in accordance with the selected
# distribution and point count, and updates the Smith chart.
async def update_inputs(app) -> None:
    
    rmax = app.r_range.value['max']
    rmin = app.r_range.value['min']
    imax = app.i_range.value['max']
    imin = app.i_range.value['min']
    
    match app.distr.value:
        case 0: #uniform
            app.zlist = ex.random_points_uniform( (rmin, rmax), (imin, imax), int(app.points.value), plot=False)
        case 1: #gaussian
            sr = ( rmax - rmin ) / 6 #Approximate one standard deviation
            mr = ( rmax + rmin ) / 2 #Calculate the mean           
            si = ( imax - imin ) / 6
            mi = ( imax + imin ) / 2
            app.zlist = ex.random_points_gaussian(mr+1j*mi, sr, si, app.corr.value, int(app.points.value), plot=False)
        case _:
            pass
    rc = [z/50 for z in app.zlist] #Normalize to 50 ohms
    app.fig.update_traces(patch = {"imag": np.imag(rc), "real": np.real(rc)}, selector = {'name':'inputs'} )
    app.plot.update()    



class App():

    def __init__(self):
        self.zlist = None
        self.setup_ui()

    def setup_ui(self):
        # Overlay element -> blocks UI during fit operaion
        with ui.element('div').style( ('position: fixed; display: block; width: 100%; height: 100%;'
                                       'top: 0; left: 0; right: 0; bottom: 0; background-color: rgba(0,0,0,0.5);'
                                       'z-index: 2; cursor: pointer;')) as self.overlay:
            with ui.element('div').classes("h-screen flex items-center justify-center"):
                ui.label('Performing fit...').style("font-size: 50px; color: white;")
        self.overlay.set_visibility(False)
    
        # ***** top-level positioning element *****
        with ui.row():            
            # ***** this is the left column *****
            with ui.column().classes('gap-2'): #control spacing between each element/panel        
                # --- input config panel ---
                ui.label('Input Config').tailwind(*section_style)
                with ui.element('div').classes('p-2 space-y-2 self-center'): #space-y works here, gap doesn't
                    with ui.row().classes('items-center'):
                        self.points = ui.input(value=points, label='Points').classes('w-32').props('square outlined dense')
                        self.distr = ui.select({0: 'Uniform', 1: 'Gaussian'},value=0,label='Distribution', 
                                          on_change=lambda x: test.set_visibility(True) if x.value==1 else test.set_visibility(False) 
                                          ).classes('w-32').props('square outlined dense')
                        ui.button('Plot', on_click=lambda: update_inputs(self)).classes('w-32')
                    
                    # real value slider
                    with ui.row().classes('items-center'):
                        ui.label('Real:')
                        self.r_range = ui.range(min=min_r, max=max_r, value={'min':10, 'max':60}).classes('w-72')
                        ui.label().bind_text_from(self.r_range, 'value',
                                      backward=lambda v: f'{v["min"]} to {v["max"]} [ohms]')
                    
                    # imag value slider
                    with ui.row().classes('items-center'):
                        ui.label('Imag:')
                        self.i_range = ui.range(min=min_i, max=max_i, value={'min':5, 'max':20}).classes('w-72')
                        ui.label().bind_text_from(self.i_range, 'value',
                                      backward=lambda v: f'{v["min"]} to {v["max"]} [ohms]')
                    
                    # corr coef slider
                    with ui.row().classes('items-center') as test:
                        ui.label('CorCoef:')
                        self.corr = ui.slider(min=-1, max=1, value=0, step=0.1).props('selection-color="transparent"').classes('w-72')
                        ui.label().bind_text_from(self.corr, 'value', backward=lambda v: f'{v}')
                        test.set_visibility(False)
                
                    # fit options
                    with ui.row().classes('items-center'):
                        ui.select(['Max SWR','Mean SWR'],value='Max SWR',label='Minimize').classes('w-32').props('square outlined dense')
                        self.levels = ui.input(value='2,3', label='Levels').classes('w-32').props('square outlined dense')
                        ui.button('Fit', on_click=lambda: fit(self)).classes('w-32')
                        
                # --- smith chart panel ---
                ui.label('Smith Chart').tailwind(*section_style)
                with ui.element('div').classes('p-2'):        
                    self.fig = go.Figure()
                    self.fig.update_layout(margin={"l":25,"r":25,"t":25,"b":25}, autosize=False, width=500, height=500)      
                    self.fig.add_trace(go.Scattersmith(smith_dict))                    
                    self.fig.add_trace(go.Scattersmith({**smith_dict, 'marker_color': 'blue', 'name': 'outputs'}))
                    self.plot = ui.plotly(self.fig) 
        
            # ***** this is the right column *****
            with ui.column().classes('gap-2'):
        
                # --- results placeholder ---
                ui.label('Fit Results').tailwind(*section_style)
                self.gridDiv = ui.element('div').style('width: 550px;').classes()
        
                # --- image placeholder --- 
                ui.label('Schematic').tailwind(*section_style)
                self.imageDiv = ui.element('div').classes('w-full').style('position: relative;')


if __name__ in {"__main__", "__mp_main__"}:    
    App()
    ui.run(port=5000, on_air=False,title='CircFit',host='0.0.0.0')