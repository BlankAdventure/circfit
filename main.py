# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 17:38:28 2024

@author: BlankAdventure
"""
import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

import circuit as ct
import experiments as ex
import plotly.graph_objects as go
from nicegui import ui
import numpy as np



# Store list if load impedances to be matched
zlist = []

max_r = 200
min_r = 0
max_i = 100
min_i = -100

opts = {'columnDefs': [{'headerName': 'Circuit', 'field': 'Circuit','sortable': False},
                       {'headerName': 'Min', 'field': 'Min','sortable': True},
                       {'headerName': 'Mean', 'field': 'Mean','sortable': True},
                       {'headerName': 'p95', 'field': 'p95','sortable': True},
                       {'headerName': 'Max', 'field': 'Max','sortable': True}],
        'rowSelection': 'single'}


def populate_table(table_data):
    if table_data is not None:
        table.clear()
        with table:
            ui.label('Fit Results')
            ui.aggrid.from_pandas(table_data, options=opts)
            #grid.on('firstDataRendered', lambda: grid.run_column_method('autoSizeAllColumns'))

def fit():
    print('doing fit')
    #print(zlist)
    res = ex.do_experiment(zlist, ex.all_components,[2],ct.cost_max_swr)
    print('DONE')
    
    df = ex.to_pandas(res)
    df = df.round(2)
    populate_table( df )

# Generates a new list of random load impedances in accordance with the selected
# distribution and point count, and updates the Smith chart.
def update() -> None:
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
            mi = (imax + imin ) / 2
            zlist = ex.random_points_gaussian(mr+1j*mi, sr, si, corr.value, int(points.value), plot=False)
        case _:
            pass
   
    rc = [z/50 for z in zlist] #Normalize to 50 ohms

    patch = dict(imag=np.imag(rc),real=np.real(rc),marker_color="red")
    fig.update_traces(patch, selector = ({'name':'inputs'}))
    plot.update()    

# Layout the UI elements    
with ui.row().classes('w-full'):    
    # ***** this is the left column *****
    with ui.column().style().classes('border bg-yellow-100 gap-2'): #control spacing between each element/panel
        with ui.element('div').classes('border p-2 bg-blue-100 space-y-2 self-center'): #space-y works here, gap doesn't
            #ui.label('***** Input Options *****')
            
            with ui.row().classes('items-center'):
                points = ui.input(value=50, label='Points').classes('w-32').props('square outlined dense')
                distr = ui.select({0: 'Uniform', 1: 'Gaussian'},value=0,label='Distribution', 
                                  on_change=lambda x: test.set_visibility(True) if x.value==1 else test.set_visibility(False) 
                                  ).classes('w-32').props('square outlined dense')
                ui.button('Plot', on_click=update).classes('w-32')
            
            with ui.row().classes('items-center'):
                ui.label('Real:')
                r_range = ui.range(min=min_r, max=max_r, value={'min':10, 'max':60}).classes('w-72')
                ui.label().bind_text_from(r_range, 'value',
                              backward=lambda v: f'{v["min"]} to {v["max"]} [ohms]')
            
            with ui.row().classes('items-center'):
                ui.label('Imag:')
                i_range = ui.range(min=min_i, max=max_i, value={'min':5, 'max':20}).classes('w-72')
                ui.label().bind_text_from(i_range, 'value',
                              backward=lambda v: f'{v["min"]} to {v["max"]} [ohms]')

            with ui.row().classes('items-center') as test:
                ui.label('CorCoef:')
                corr = ui.slider(min=-1, max=1, value=0, step=0.1).props('selection-color="transparent"').classes('w-72')
                ui.label().bind_text_from(corr, 'value', backward=lambda v: f'{v}')
                test.set_visibility(False)
        

            with ui.row().classes('items-center'):
                ui.select(['Max SWR','Mean SWR'],value='Max SWR',label='Minimize').classes('w-32').props('square outlined dense')
                ui.input(value='2,3', label='Levels').classes('w-32').props('square outlined dense').disable()
                ui.button('Fit', on_click=fit).classes('w-32')
                
        with ui.element('div').classes('border p-2 bg-blue-100'):        
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
                marker_symbol='circle',
                marker_size=8,
                marker_color="blue",
                mode='markers',
                showlegend=False,
                name='outputs'
            ))        
            plot = ui.plotly(fig) 

    # ***** this is the right column *****
    with ui.column().style().classes('border bg-yellow-100 gap-2 w-auto'):        
        # Results table
        with ui.element('div').classes('border p-2 bg-blue-100 space-y-2 self-center').style('width: 550px;') as table:
            ui.label('Fit Results')
            populate_table(None)
        # Circuit image:    
        with ui.element('div').classes('border p-2 bg-blue-100'): 
            ui.label('circuit diagram')




ui.run()