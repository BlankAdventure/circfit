# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 17:38:28 2024

@author: BlankAdventure
"""
#from matplotlib import pyplot as plt
import plotly.graph_objects as go
from nicegui import ui
import numpy as np

zlist = []


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
        case _:
            pass
   
    zlist = rr + 1j*ri
    rc = [z/50 for z in zlist]

    patch = dict(imag=np.imag(rc),real=np.real(rc),marker_color="red")
    fig.update_traces(patch, selector = ({'name':'inputs'}))
    plot.update()    

    
with ui.row().classes('w-full'):    
    with ui.column().style().classes('border bg-yellow-100'): #LEFT COLUMN
        with ui.element('div').classes('border p-2 bg-blue-100'):        
            ui.label('***** Input Options *****')
            
            with ui.row().classes('items-center'):
                counts = ui.input(value=50, label='Points').classes('w-32').props('square outlined dense')
                distr = ui.select(['Uniform','Gaussian'],value='Uniform',label='Distribution').classes('w-32').props('square outlined dense')
            
            with ui.row().classes('items-center'):
                ui.label('Real:')
                r_range = ui.range(min=0, max=200, value={'min':10, 'max':60}).classes('w-72')
                ui.label().bind_text_from(r_range, 'value',
                              backward=lambda v: f'{v["min"]} to {v["max"]} [ohms]')
            
            with ui.row().classes('items-center'):
                ui.label('Imag:')
                i_range = ui.range(min=-100, max=100, value={'min':5, 'max':20}).classes('w-72')
                ui.label().bind_text_from(i_range, 'value',
                              backward=lambda v: f'{v["min"]} to {v["max"]} [ohms]')
            with ui.row():
                ui.button('Plot', on_click=update).classes('w-32')
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
            # print(fig)
            plot = ui.plotly(fig) #.classes('w-64 h-64')
            #patch = dict(imag=[2,4],real=[2,3],marker_color="red")
            #fig.update_traces(patch, selector = ({'name':'inputs'}))
            #plot.update()

    with ui.column():        
        ui.label('***** Right Column *****')




#    with ui.element('div').classes('border p-2 bg-blue-100'):
#        ui.label('LR - inside a colored div')

    ##ui.label('UL')
    #ui.label('UR')
    #ui.label('LL')
    #ui.label('LR')
    #ui.card()
    #ui.card()
    #ui.card()
    #ui.card()
    
ui.run()