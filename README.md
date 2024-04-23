# circfit
Multiple impedance circuit fitting experiments

The usual matching problem is presented as having some network that is a function of frequency, H(w), subject to load impedance Z, and the goal is to optimize H(w) to obtain the best possible match over some frequency range (i.e., bandwidth). However, there is another form of this problem that I have often encountered in devices driving or reading from sensors. Here the system typically operates at some fixed frequency and the sensors present a range of imepdances to the circuit. Thus we want to find a matching circuit topology that best matches the impedance distribution, subject to a fixed frequency. The code in this repo is for exploring this problem.

 **----- circuit.py (Basic Usage) -----**

This module provides basic functionality for generating circuit networks and finding optimal component values given a set of input/load impedances.


The following code instansiates a new Circuit object then adds series capacitor and parallel inductor, *after* the load. Component values are not specified at this point, as the object of the code is to find optimal values based on a set of input/load impances.
```python
import circuit as ct

# Create a new circuit object   
c = ct.Circuit()

# Add two components. Values are not specified as the goal is to find optimal
# values later.
c.add_series_c()
c.add_parallel_l()
```

Next, we'll specify 3 load impedances and perform the fit:

```python
# Optimize across 3 different load impedances
zlist = [30+1j*10, 20+1j*5, 35+1j*15]

# Do the fit. Must specifcy initial reactance values for the two components
res = c.fit(zlist, x0=[-20,20], cost_func=ct.cost_mean_swr)
```

The user must specify initial values for the reactances, listed in the order they were added. The cost function has been set to cost_mean_swr, which seeks to minimize the mean SWR across the set of impedances. Once complete, we can display the final circuit using:

```python
# Draw the circuit, showing reactances
c.draw()
```
<img src="https://github.com/BlankAdventure/circfit/assets/24900496/de8cdac7-5ef9-407b-ba25-fb73c840e445" height="240">

Specify a frequency value to convert from reactances to circuit component values:

```python
c.draw(F=1e6)
```
<img src="https://github.com/BlankAdventure/circfit/assets/24900496/ede28df3-aeff-418c-b37c-a0525e79583d" height="240">

Finally, we can examine how well the circuit performs:
```python
# Get the matched or input impedances 
zm = c.get_zin(zlist)

# Compare the SWRs 
swr_in = ct.swr_from_z(zlist)
swr_out = ct.swr_from_z(zm)
print(swr_in)
print(swr_out)
````
```
[1.77, 2.53, 1.65]
[1.00, 1.57, 1.24]
```
Notice the improved match across all 3 loads.

 **----- main.py -----**
 
This module provides a web UI (via nicegui) for more advanced optimizations. In the above example, we specified a fixed network and found optimal values for it. However, there could be better networks available. The web UI, via experiments.py, enumerates all possible network topologies* and optimizes across them. 

An example run can be seen below:
<img src="https://github.com/BlankAdventure/circfit/assets/24900496/def0db54-3d01-473a-bc3a-973996db43d1" height="600">

Here, 50 load impedances were simulated. Values were generated from a *Uniform* probability distribution (other distributions can be choosen) whose real and imaginary components span the range specfied by the *Real* and *Imag* sliders respecitvely. The resulting impedances can be seen plotted in red on the Smith chart. We've chosen to minimize the *Max SWR* across the range of impedances (others can be choosen). *Levels* specifies the number of circuit components to use - in this case 2 and 3. Once the *Fit* has completed, the table on the right appears. It lists all the possible circuit topologies limited to 2 and 3 components**. Columns show the minimum, mean, 95th-percentile and max value of the resulting matched set of SWRs. Each row can be clicked on and the Smith chart will update to show the matched impedances (blue dots), and, a circuit diagram will appear.

In this caese, notice that the selected 2-component circuit performed the same as the more complicated 3-component circuits. 

*Currently, only networks following a repeated series-shunt topology are generated.

**Combinations consisting of back-to-back arrangements of same-orientated componnets (i.e., Circuit.add_series_X() followed by Circuit.add_series_X() or Circuit.add_parallel_X() followed by Circuit.add_parallel_X() ) are excluded -- exercise for the reader as to why :)



 

