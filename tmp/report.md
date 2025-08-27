# MEA-data analysis
(see [[protocol-description]])

The purpose of this analysis is to take a closer look at the MEA data and compare the results of the 3 experimental conditions:

**C1:** Control 
**C2:** Scramble-expressing Chronic GR activation
**C3:** miR-186-5p inhibitor Chronic GR activation

This report provides an example of the types of analyses that can be performed upon further discussion. Note that these results are a demonstration and they are not entirely reliable at this stage nor have they been properly interpreted. There are several factors that need to be accounted for, particularly:
- These metrics are designed to capture properties of networks, but each recording comprises multiple wells, so multiple unconnected sub-populations. To do the analysis properly, we need to treat each sub-population (16 electrodes) as an individual / independent network 
- The manifold extraction examples were ran on a sub-set of data (5 to 10 sec) and running it on the full dataset would require more RAM capacity (ran on a laptop with 32GB)

Nevertheless, the analysis is implemented and can be applied and modified as needed. We should meet and discuss how to proceed from here. All code used in this analysis is provided in the attached file and the individual analyses reported on below can be replicated with the corresponding JuPyTer notebooks (.ipynb files). 

---
### Data Loading
All the analyses were performed in Python, but the data was acquired and stored via taylored software from Axion, in MatLab.

Using [AxionFileLoader](https://github.com/axionbio/AxionFileLoader) in Matlab, I read in the files provided (raw spk data), using:
```Matlab
[Electrodes, Times] = AxisFile('../data/n3-DIV17-01.spk').SpikeData.LoadAllSpikes;
Channels = Electrodes.Channel;
Chamber = Electrodes.Achk;
save("../data/n3-DIV17-01.mat");
```

Having the data in .mat format, I can now load it into python and, using the channel id and spike times, convert it into a `SpikeList` object (from a software package I developed that has pre-built a lot of useful functionality for the analysis of spiking data).

```Python
# Load datasets  
data = loadmat('../data/n1-DIV17-01.mat')  
ids = data['Channels'][0]  
times = data['Times'][0]  
spk_times = [(i, times[idx]*1000) for idx, i in enumerate(ids)]  
spk_ids = np.unique(ids)  
sl_n1 = SpikeList(spk_times, spk_ids)
```

---
### Recorded Spiking Activity
(see [[raw_spikes.ipynb]])
The simplest and most straightforward analysis is to just look at the raw activity, visualize the differences and quantify. Before going any further, let's compare the raw spiking activity in the three conditions (for visualization purposes, we plot only the first 5 seconds of activity):
![[n1-small-raster.png]]
![[n2-small-raster.png]]
![[n3-small-raster.png]]

---
### Spiking statistics
(see [[global-spiking-stats.ipynb]])

There are a lot of statistical analysis to quantify the properties of spiking activity. We apply a battery of metrics (as implemented in (conic-tools)) which we split into:
- Activity -> rates, counts, ...
- Regularity -> CV-ISI, LV-ISI, LVR-ISI, entropy, ...
- Synchrony -> Pearson pairwise CC, ISI distance, SPIKE Distance, ... (see Kreuz, 2017)
![[Research/MEA-data/code/PyDataAnalysis/plots/activity.png]]
![[regularity.png]]
![[sync.png]]

These stats can also be summarised and compared with simpler and more intuitive plots, 

**Note:** a prominent feature of cultured networks are spontaneous network bursts. In this report, I do not include a detailed analysis on the statistics of NBs because I did not have these analyses readily available. Nevertheless, these are part of the standard toolkit provided by the Axion Biosystems software.

---
### Burst statistics
These were part of the original manuscript. I did not have a usable implementation, given that these are very specific to MEA cultures. I copied the figure below for us to discuss as it appears to be inconsistent with the data I plotted above..
![[Pasted image 20240326211747.png]]

---
### Population states and geometry of population dynamics
See complete report in [[state-manifolds-info]]

---
# Suggestions for future work

We can implement and use a computational model of population dynamics, explicitly fitted to the MEA recordings and designed to structurally and functionally mirror the plated networks (accounting for neuron numbers, architecture and any mechanisms of interest). This model can then be used to infer causal relations and test relevant hypotheses (see [1] for a good example):

![[Pasted image 20240326215116.png]]

- Fit population model and verify how the parameters vary across different conditions:
  - simplified balanced rate model
  - simplified balanced random network of spiking neurons
  - spiking neural network with conductance-based synapses accounting for detailed Glu/GABA transmission and their variations 

The model is characterised by a set of parameters of interest $\Theta = \{ \gamma, w, \bar{g}_{\mathrm{AMPA}}...\}$, which can be learned / optimized, e.g., via gradient descent or brute-force to minimize a cost function based on the observed population dynamics: $\mathcal{L}_{\Theta} = \sum_{i}(s^{i}_{d} - s^{i}_{m}(\theta))$. The value of the optimized parameter set and the learning trajectory informs us of the role of certain features of interest. The resulting best fit tells us the magnitude of model parameters.
- For example, the model can test the important and impact of the experimental manipulations on GABAergic and Glutamatergic transmission
- E/I balance and PSC amplitudes
- (...)


We could start by setting up the tools (software) to do this systematically (take in recorded data and do multi-objective optimization). I have work in progress for other projects that can be of use for this purpose.

#### References:
[1] Pachitaru, M., Stringer, C., Okun, M., Bartho, P., Harris, K., Latham, P., Sahani, M. & Lesica, N. (2016). Inhibitory control of shared variability
in cortical networks. bioRXiv. http://dx.doi.org/10.1101/041103

