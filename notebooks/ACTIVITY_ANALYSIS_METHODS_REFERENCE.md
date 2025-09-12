# Comprehensive Activity Analysis Methods Reference

> **Methods Section for Research Papers**
> 
> **Neural Activity Analysis in Multi-Electrode Array Recordings**
> 
> Neural activity patterns were quantified using a comprehensive suite of electrophysiological metrics implemented in MEA-Flow. Basic activity measures included firing rates, spike counts, and population-level statistics computed across all active electrodes (≥10 spikes minimum threshold). Temporal regularity was assessed using coefficient of variation of inter-spike intervals (CV-ISI), local variation (LV), revised local variation accounting for refractory periods (LVR), and Shannon entropy of ISI distributions (Shinomoto et al., 2003; Ponce-Alvarez et al., 2010). Population dynamics were characterized through Fano factor analysis (variance-to-mean ratio of spike counts), participation ratio (fraction of electrodes active per time bin), and population vector length calculations.
> 
> Network synchronization was quantified using pairwise Pearson correlation coefficients of binned spike trains (10ms bins) and advanced spike train distance metrics including ISI-distance, SPIKE-distance, and SPIKE-synchrony measures from the PySpike library (Kreuz et al., 2013; Mulansky & Kreuz, 2016). Van Rossum distances were computed using exponential kernels (τ = 20ms) to assess spike train similarity (van Rossum, 2001). Population-level synchrony was evaluated through chi-square distance from independence, population spike synchrony (mean pairwise coincidence rates), and synchrony indices quantifying multi-electrode co-activation patterns.
> 
> Burst detection employed dual-threshold algorithms for individual electrodes (maximum ISI = 100ms, minimum 5 spikes per burst) and network-level burst identification using envelope methods with adaptive thresholds (mean + 1.25×SD of population activity; Chiappalone et al., 2006; Wagenaar et al., 2006). Network bursts required ≥35% electrode participation and minimum 50ms duration. All metrics were computed using standardized time bins with appropriate refractory period corrections and validated against established MEA analysis protocols.

## Overview

This document provides comprehensive mathematical descriptions of all neural activity analysis methods implemented in MEA-Flow. Methods are organized by functional category with formal equations, biological interpretations, and computational considerations for multi-electrode array (MEA) data analysis.

## 1. Basic Activity Metrics

### 1.1 Firing Rate Analysis

#### 1.1.1 Individual Channel Firing Rate
**Mathematical Definition:**
```
FR_i = N_i / T
```
Where:
- `FR_i` = firing rate of channel i (Hz)
- `N_i` = total spike count for channel i
- `T` = recording duration (seconds)

**Population Statistics:**
```
FR_mean = (1/n) × Σ(i=1 to n) FR_i
FR_std = sqrt((1/n) × Σ(i=1 to n) (FR_i - FR_mean)²)
FR_median = median(FR_1, FR_2, ..., FR_n)
```

#### 1.1.2 Network Firing Rate
**Mathematical Definition:**
```
NFR = (Σ(i=1 to n) N_i) / T = Total_Spikes / T
```

**Biological Significance:** Measures overall network excitability and activity level. Higher NFR indicates increased network drive or reduced inhibition.

### 1.2 Spike Count Statistics

#### 1.2.1 Total and Per-Channel Spike Counts
```
Total_Spikes = Σ(i=1 to n) N_i
Mean_Spike_Count = (1/n) × Σ(i=1 to n) N_i
Std_Spike_Count = sqrt((1/n) × Σ(i=1 to n) (N_i - Mean_Spike_Count)²)
```

#### 1.2.2 Activity Fraction
```
Activity_Fraction = n_active / n_total
```
Where `n_active` = number of channels with ≥ minimum spike threshold.

### 1.3 Population Vector Analysis

#### 1.3.1 Population Vector Length
For time-binned spike matrix **S** (n_channels × n_bins):
```
PVL(t) = ||S(:,t)||₂ = sqrt(Σ(i=1 to n) S(i,t)²)
PVL_mean = (1/T_bins) × Σ(t=1 to T_bins) PVL(t)
```

**Biological Significance:** Measures coordinated population activity strength. Higher PVL indicates more synchronized network states.

#### 1.3.2 Participation Ratio
```
PR(t) = (Σ(i=1 to n) I[S(i,t) > 0]) / n
PR_mean = (1/T_bins) × Σ(t=1 to T_bins) PR(t)
PR_std = sqrt((1/T_bins) × Σ(t=1 to T_bins) (PR(t) - PR_mean)²)
```
Where `I[·]` is the indicator function.

**Biological Significance:** Quantifies how evenly distributed activity is across electrodes. Low PR indicates sparse, localized activity; high PR indicates widespread synchronization.

## 2. Regularity and Temporal Pattern Analysis

### 2.1 Inter-Spike Interval (ISI) Based Metrics

#### 2.1.1 Coefficient of Variation (CV-ISI)
**Mathematical Definition:**
```
CV_ISI = σ_ISI / μ_ISI
```
Where:
- `σ_ISI` = standard deviation of ISIs
- `μ_ISI` = mean ISI

**Interpretation:**
- CV ≈ 0: Regular firing (clock-like)
- CV ≈ 1: Poisson-like firing
- CV > 1: Irregular/bursty firing

#### 2.1.2 Local Variation (LV)
**Mathematical Definition:**
```
LV = (3/(n-1)) × Σ(i=1 to n-1) ((ISI_i - ISI_{i+1})² / (ISI_i + ISI_{i+1})²)
```

**Biological Significance:** Measures local irregularity by comparing adjacent ISIs. More sensitive to short-term variations than CV.

#### 2.1.3 Revised Local Variation (LVR)
**Mathematical Definition:**
```
ISI_corrected = ISI - τ_refrac
LVR = LV(ISI_corrected[ISI_corrected > 0])
```
Where `τ_refrac` = refractory period (typically 1ms).

**Purpose:** Accounts for absolute refractory period effects on ISI distributions.

#### 2.1.4 ISI Entropy
**Mathematical Definition:**
```
H_ISI = -Σ(j=1 to k) p_j × log(p_j)
```
Where `p_j` is the probability of ISI bin j from histogram with k bins.

**Biological Significance:** Higher entropy indicates more irregular, unpredictable spike timing patterns.

### 2.2 Burstiness and Memory Measures

#### 2.2.1 Burstiness Index
**Mathematical Definition:**
```
B = (σ_ISI - μ_ISI) / (σ_ISI + μ_ISI)
```
**Range:** -1 (perfectly regular) to +1 (maximally bursty)

#### 2.2.2 Memory Coefficient
**Mathematical Definition:**
```
M = corr(ISI_i, ISI_{i+1}) = Σ((ISI_i - μ)(ISI_{i+1} - μ)) / sqrt(Σ(ISI_i - μ)² × Σ(ISI_{i+1} - μ)²)
```

**Biological Significance:** Measures correlation between consecutive ISIs, indicating temporal memory in spike generation.

### 2.3 Population-Level Regularity

#### 2.3.1 Fano Factor
**Mathematical Definition:**
For spike counts in time bins:
```
FF_i = Var(Counts_i) / Mean(Counts_i)
FF_population = (1/n) × Σ(i=1 to n) FF_i
```

**Interpretation:**
- FF = 1: Poisson process
- FF < 1: Sub-Poisson (regular)
- FF > 1: Super-Poisson (irregular/bursty)

#### 2.3.2 Population Coefficient of Variation
```
Population_CV = σ_pop / μ_pop
```
Where `σ_pop` and `μ_pop` are standard deviation and mean of total population activity per time bin.

## 3. Synchrony and Correlation Analysis

### 3.1 Pairwise Correlation Measures

#### 3.1.1 Pearson Cross-Correlation
For binned spike trains **X** and **Y**:
```
ρ_XY = Σ((X_t - μ_X)(Y_t - μ_Y)) / sqrt(Σ(X_t - μ_X)² × Σ(Y_t - μ_Y)²)
```

**Population Statistics:**
```
ρ_mean = (2/(n(n-1))) × Σ(i<j) ρ_ij
ρ_std = sqrt((2/(n(n-1))) × Σ(i<j) (ρ_ij - ρ_mean)²)
```

### 3.2 Spike Train Distance Measures

#### 3.2.1 Van Rossum Distance
**Mathematical Definition:**
```
D_VR = sqrt(∫_{-∞}^{∞} (f_X(t) - f_Y(t))² dt)
```
Where filtered spike trains are:
```
f_X(t) = Σ_i exp(-(t - t_i^X)/τ) × H(t - t_i^X)
```
`H(·)` = Heaviside step function, `τ` = time constant (typically 20ms)

#### 3.2.2 ISI-Distance (PySpike)
**Mathematical Definition:**
```
D_ISI(t) = |ν_X(t) - ν_Y(t)| / max(ν_X(t), ν_Y(t))
```
Where `ν_X(t)` is the instantaneous ISI-based firing rate.

#### 3.2.3 SPIKE-Distance (PySpike)
**Mathematical Definition:**
```
D_SPIKE(t) = |S_X(t) - S_Y(t)| / max(S_X(t), S_Y(t))
```
Where `S_X(t)` represents spike train X as a continuous function.

### 3.3 Population Synchrony Measures

#### 3.3.1 Chi-Square Distance from Independence
**Mathematical Definition:**
```
χ² = (1/T_bins) × Σ(t=1 to T_bins) ((O_t - E_t)² / E_t)
```
Where:
- `O_t` = observed number of active channels at time t
- `E_t` = expected number under independence assumption

#### 3.3.2 Population Spike Synchrony
**Mathematical Definition:**
```
PSS = (2/(n(n-1))) × Σ(i<j) (C_ij / A_ij)
```
Where:
- `C_ij` = number of bins where both channels i and j spike
- `A_ij` = number of bins where either channel i or j spikes

#### 3.3.3 Synchrony Index
**Mathematical Definition:**
```
SI = (Number of bins with >1 active channel) / Total_bins
```

## 4. Burst Detection and Analysis

### 4.1 Individual Channel Burst Detection

#### 4.1.1 ISI-Based Burst Detection Algorithm
**Criteria:**
1. Maximum ISI within burst: `ISI_max` (default: 100ms)
2. Minimum spikes per burst: `N_min` (default: 5)

**Algorithm:**
```
For each ISI_i in spike train:
    if ISI_i ≤ ISI_max:
        if not in_burst:
            start_burst(i)
        continue_burst()
    else:
        if in_burst and spike_count ≥ N_min:
            end_burst(i-1)
```

#### 4.1.2 Burst Statistics
**Duration:**
```
Burst_Duration = t_end - t_start
```

**Intra-Burst Firing Rate:**
```
Burst_Rate = N_spikes / Burst_Duration
```

**Inter-Burst Intervals:**
```
IBI_i = Burst_Start_{i+1} - Burst_End_i
```

### 4.2 Network Burst Detection

#### 4.2.1 Envelope Algorithm
**Step 1: Population Activity Envelope**
```
E(t) = Σ(i=1 to n) S_i(t)    [spike count per time bin]
```

**Step 2: Smoothing (Moving Average)**
```
E_smooth(t) = (1/w) × Σ(k=-w/2 to w/2) E(t+k)
```

**Step 3: Adaptive Threshold**
```
Threshold = μ_E + α × σ_E
```
Where `α` = threshold factor (default: 1.25)

**Step 4: Network Burst Criteria**
- Duration: `t_end - t_start ≥ t_min` (default: 50ms)
- Participation: `N_participating / N_total ≥ f_min` (default: 0.35)

#### 4.2.2 Network Burst Characteristics

**Participation Ratio:**
```
PR_burst = N_participating_electrodes / N_total_electrodes
```

**Peak Activity:**
```
Peak_Activity = max(E(t)) for t ∈ [t_start, t_end]
```

**Total Spikes:**
```
Total_Spikes = Σ(t=t_start to t_end) E(t)
```

**Inter-Network-Burst Intervals:**
```
INBI_i = NB_Start_{i+1} - NB_End_i
```

## 5. Advanced Population Measures

### 5.1 Global Activity Fluctuations

#### 5.1.1 Coefficient of Variation of Global Activity
```
CV_global = σ_global / μ_global
```
Where global activity = `Σ(i=1 to n) S_i(t)` per time bin.

### 5.2 Spatial Activity Patterns

#### 5.2.1 Active Channel Count per Time Bin
```
Active_Count(t) = Σ(i=1 to n) I[S_i(t) > 0]
```

#### 5.2.2 Activity Spread Measures
**Participation Ratio Variability:**
```
PR_variability = σ_PR / μ_PR
```

## 6. Computational Considerations

### 6.1 Time Binning Parameters
- **Correlation Analysis:** 10ms bins (balances temporal resolution vs. statistical power)
- **Population Measures:** 1ms bins (captures fine temporal dynamics)
- **Burst Detection:** 10ms bins (envelope smoothing)

### 6.2 Statistical Thresholds
- **Minimum Spikes for Rate:** 10 spikes per channel
- **Minimum ISI Samples:** 5 intervals for regularity metrics
- **Maximum Pairs for Synchrony:** 500 pairs (computational efficiency)

### 6.3 Refractory Period Corrections
- **Standard Refractory Period:** 1ms
- **Applied to:** LVR calculations, burst detection algorithms

## 7. Metric Interpretation Guidelines

### 7.1 Activity Level Classification
- **Low Activity:** Mean FR < 0.1 Hz
- **Moderate Activity:** 0.1 Hz ≤ Mean FR < 1 Hz  
- **High Activity:** Mean FR ≥ 1 Hz

### 7.2 Regularity Classification
- **Regular:** CV < 0.5, LV < 0.5
- **Irregular:** 0.5 ≤ CV < 1.5, 0.5 ≤ LV < 1.5
- **Bursty:** CV > 1.5 or LV > 1.5

### 7.3 Synchrony Level Classification
- **Low Synchrony:** Mean ρ < 0.1
- **Moderate Synchrony:** 0.1 ≤ Mean ρ < 0.3
- **High Synchrony:** Mean ρ ≥ 0.3

## References

Chiappalone, M., Bove, M., Vato, A., Tedesco, M., & Martinoia, S. (2006). Dissociated cortical networks show spontaneously correlated activity patterns during in vitro development. *Brain Research*, 1093(1), 41-53.

Kreuz, T., Chicharro, D., Houghton, C., Andrzejak, R. G., & Mormann, F. (2013). Monitoring spike train synchrony. *Journal of Neurophysiology*, 109(5), 1457-1472.

Mulansky, M., & Kreuz, T. (2016). PySpike—A Python library for analyzing spike train synchrony. *SoftwareX*, 5, 183-189.

Ponce-Alvarez, A., Thiele, A., Albright, T. D., Stoner, G. R., & Deco, G. (2013). Stimulus-dependent variability and noise correlations in cortical MT neurons. *Proceedings of the National Academy of Sciences*, 110(32), 13162-13167.

Shinomoto, S., Shima, K., & Tanji, J. (2003). Differences in spiking patterns among cortical neurons. *Neural Computation*, 15(12), 2823-2842.

van Rossum, M. C. (2001). A novel spike distance. *Neural Computation*, 13(4), 751-763.

Wagenaar, D. A., Pine, J., & Potter, S. M. (2006). An extremely rich repertoire of bursting patterns during the development of cortical cultures. *BMC Neuroscience*, 7(1), 11.
