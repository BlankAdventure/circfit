# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 17:38:28 2024

@author: BlankAdventure
"""
import plotly.graph_objects as go
from nicegui import ui
import numpy as np

# Store list if load impedances to be matched
zlist = []

max_r = 200
min_r = 0
max_i = 100
min_i = -100


# Generates a new list of random load impedances in accordance with the selected
# distribution and point count, and updates the Smith chart.
def update() -> None:
    match distr.value:
        case 'Uniform':
            rr = np.random.uniform(low=r_range.value['min'],high=r_range.value['max'],size=int(counts.value))
            ri = np.random.uniform(low=i_range.value['min'],high=i_range.value['max'],size=int(counts.value))
        case 'Gaussian':
            sr = ( r_range.value['max'] - r_range.value['min'] ) / 6
            mr = ( r_range.value['max'] + r_range.value['min'] ) / 2            
            si = ( i_range.value['max'] - i_range.value['min'] ) / 6
            mi = ( i_range.value['max'] + i_range.value['min'] ) / 2
            rr = np.random.normal(mr, sr, int(counts.value))
            ri = np.random.normal(mi, si, int(counts.value))            
            rr = rr[(rr >= min_r) & (rr <= max_r)]
            ri = ri[(ri >= min_i) & (ri <= max_i)]
        case _:
            pass
   
    zlist = rr + 1j*ri #Convert to complex impedance
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
                counts = ui.input(value=50, label='Points').classes('w-32').props('square outlined dense')
                distr = ui.select(['Uniform','Gaussian'],value='Uniform',label='Distribution').classes('w-32').props('square outlined dense')
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

            with ui.row().classes('items-center'):
                ui.select(['Max SWR','Mean SWR'],value='Max SWR',label='Minimize').classes('w-32').props('square outlined dense')
                ui.input(value='2,3', label='Levels').classes('w-32').props('square outlined dense').disable()
                ui.button('Fit').classes('w-32')
                
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
    with ui.column():        
        ui.label('***** Right Column *****')





ui.run()