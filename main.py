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
from nicegui import ui, Tailwind
import numpy as np
from functools import wraps, partial

zlist = [] # Store list of load impedances to be matched/fit

# min/max settings for sliders/impedance values
max_r = 200
min_r = 0
max_i = 100
min_i = -100
points = 5

#section_style = Tailwind().text_color('white').font_weight('bold').background_color('black').padding(0).width('full')
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
def refresh_table(res_df) -> None:
    def update_outputs(event) -> None:
        def update_dwg(event) -> None:            
            if freq.value is not None:
                #print(freq.value)
                h.set_content(cm.draw(for_web=True,F=float(freq.value)).decode('utf-8'))
            else:
                h.set_content(cm.draw(for_web=True).decode('utf-8'))

        idx = int(event.args['rowId'])
        cm = res_df.loc[idx, 'Model']
        zin = cm.get_zin(zlist)
        rc = [z/50 for z in zin] #Normalize to 50 ohms
        fig.update_traces(patch=dict(imag=np.imag(rc),real=np.real(rc)), selector = ({'name':'outputs'}))
        plot.update()    
        
        imageDiv.clear()
        with imageDiv:            
            freq = ui.input(value=None,placeholder='<none>',label='Freq [Hz]').classes('w-28 bg-white').props(
                'clearable square outlined dense input-class="font-mono" stack-label').style(
                    'position: absolute; z-index: 1; top: -8px; right: 0px;').on('blur', lambda e: update_dwg(e))
                    
            with ui.html().classes('max-w-max').style('margin: 0 auto; justify-content: center;') as h:
                update_dwg(None)
       
            
            
    if res_df is not None:
        subset = res_df.loc[:, res_df.columns != 'Model'] #We only want the numerical results data, not the objects column
        x = len(res_df)
        if x > 18:
            style =  f'height: {(18+1)*25 + 15}px;'
        else:
            style =  f'height: {(x+1)*25 + 15}px;'

        gridDiv.clear()
        with gridDiv:
            ui.aggrid.from_pandas(subset, 
                        options=table_options).on('rowDoubleClicked', 
                        lambda event: update_outputs(event) , ['rowId'] ).classes().style(style)
                                                  
     
# Do the fit!
@wrap
def fit() -> None:
    if zlist:
        overlay.set_visibility(True)
        depth = [int(i.strip()) for i in levels.value.split(',')]
        res = ex.do_experiment(zlist, ex.all_components,depth,ct.cost_max_swr)
        refresh_table( ex.to_pandas(res, include_model=True).round(2) )
        overlay.set_visibility(False)


    
# Generates a new list of random load impedances in accordance with the selected
# distribution and point count, and updates the Smith chart.
def update_inputs() -> None:
    global zlist
    rmax = r_range.value['max']
    rmin = r_range.value['min']
    imax = i_range.value['max']
    imin = i_range.value['min']
    
    match distr.value:
        case 0: #uniform
            zlist = ex.random_points_uniform( (rmin, rmax), (imin, imax), int(points.value), plot=False)
        case 1: #guassian
            sr = ( rmax - rmin ) / 6
            mr = ( rmax + rmin ) / 2            
            si = ( imax - imin ) / 6
            mi = ( imax + imin ) / 2
            zlist = ex.random_points_gaussian(mr+1j*mi, sr, si, corr.value, int(points.value), plot=False)
        case _:
            pass
    rc = [z/50 for z in zlist] #Normalize to 50 ohms
    fig.update_traces(patch = dict( imag=np.imag(rc), real=np.real(rc) ), selector = ({'name':'inputs'}))
    plot.update()    



# **************************************************
#           Layout the UI elements                 #
# **************************************************


# Overlay element -> blocks UI during fit operaion
with ui.element('div').style('position: fixed; display: block; width: 100%; height: 100%; top: 0; left: 0; right: 0; bottom: 0; background-color: rgba(0,0,0,0.5); z-index: 2; cursor: pointer;') as overlay:
    with ui.element('div').classes("h-screen flex items-center justify-center"):
        ui.label('Performing fit...').style("font-size: 50px; color: white;")
overlay.set_visibility(False)

# ***** top-level positioning element *****
with ui.row().classes():
#with ui.element('div').classes('border bg-red-50 p-2 m-2 gap-2').style('display: flex; width: 100%;'): 
    
    # ***** this is the left column *****
    with ui.column().classes('gap-2'): #control spacing between each element/panel

        # --- input config panel ---
        ui.label('Input Config').tailwind(*section_style)
        with ui.element('div').classes('p-2 space-y-2 self-center'): #space-y works here, gap doesn't
            with ui.row().classes('items-center'):
                points = ui.input(value=points, label='Points').classes('w-32').props('square outlined dense')
                distr = ui.select({0: 'Uniform', 1: 'Gaussian'},value=0,label='Distribution', 
                                  on_change=lambda x: test.set_visibility(True) if x.value==1 else test.set_visibility(False) 
                                  ).classes('w-32').props('square outlined dense')
                ui.button('Plot', on_click=update_inputs).classes('w-32')
            
            # real value slider
            with ui.row().classes('items-center'):
                ui.label('Real:')
                r_range = ui.range(min=min_r, max=max_r, value={'min':10, 'max':60}).classes('w-72')
                ui.label().bind_text_from(r_range, 'value',
                              backward=lambda v: f'{v["min"]} to {v["max"]} [ohms]')
            
            # imag value slider
            with ui.row().classes('items-center'):
                ui.label('Imag:')
                i_range = ui.range(min=min_i, max=max_i, value={'min':5, 'max':20}).classes('w-72')
                ui.label().bind_text_from(i_range, 'value',
                              backward=lambda v: f'{v["min"]} to {v["max"]} [ohms]')
            
            # corr coef slider
            with ui.row().classes('items-center') as test:
                ui.label('CorCoef:')
                corr = ui.slider(min=-1, max=1, value=0, step=0.1).props('selection-color="transparent"').classes('w-72')
                ui.label().bind_text_from(corr, 'value', backward=lambda v: f'{v}')
                test.set_visibility(False)
        
            # fit options
            with ui.row().classes('items-center'):
                ui.select(['Max SWR','Mean SWR'],value='Max SWR',label='Minimize').classes('w-32').props('square outlined dense')
                levels = ui.input(value='2,3', label='Levels').classes('w-32').props('square outlined dense')
                ui.button('Fit', on_click=fit).classes('w-32')
                
        # --- smith chart panel ---
        ui.label('Smith Chart').tailwind(*section_style)
        with ui.element('div').classes('p-2'):        
            fig = go.Figure()
            fig.update_layout(margin=dict(l=25, r=25, t=25, b=25), autosize=False, width=500, height=500)        
    
            fig.add_trace(go.Scattersmith(
                imag=[],
                real=[],
                opacity = 0.7,
                marker_symbol='circle',
                marker_size=8,
                marker_color='red',
                mode='markers',
                showlegend=False,
                name='inputs'
            ))
            fig.add_trace(go.Scattersmith(
                imag=[],
                real=[],
                opacity = 0.7,
                marker_symbol='circle',
                marker_size=8,
                marker_color="blue",
                mode='markers',
                showlegend=False,
                name='outputs'
            ))        
            plot = ui.plotly(fig) 

    # ***** this is the right column *****
    with ui.column().style().classes('gap-2'):

        # --- results placeholder ---
        ui.label('Fit Results').tailwind(*section_style)
        gridDiv = ui.element('div').style('width: 550px;').classes()

        # --- image placeholder --- 
        #with ui.row().classes('border justify-between w-full'):
        ui.label('Schematic').tailwind(*section_style)
        imageDiv = ui.element('div').classes('w-full').style('position: relative;')



ui.run(port=5000, on_air=False,title='CircFit',host='0.0.0.0')