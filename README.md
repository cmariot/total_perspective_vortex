# total_perspective_vortex
This project aims to create a brain computer interface based on electroencephalographic data (EEG data) with the help of machine learning algorithms.

### Installation

Clone the project on your local machine:

```bash
git clone https://github.com/cmariot/total_perspective_vortex.git
cd total_perspective_vortex
```

Download the dataset from [physionet](https://physionet.org/content/eegmmidb/1.0.0/) and extract it in the `dataset` folder.

You need to have [conda](https://docs.conda.io/en/latest/) installed on your system. Then you can create a new conda environment with the following command:

```bash
conda install --channel=conda-forge --name=base mamba
mamba create --override-channels --channel=conda-forge --name=total_perspective_vortex mne
conda activate total_perspective_vortex
```
### Dataset

The data used in this project come from EEG recordings, obtained from 109 volunteers. The volunteers were told to perform 4 different mental tasks, while their brain activity was recorded by 64 electrodes. The data is available at [physionet](https://physionet.org/content/eegmmidb/1.0.0/).

<div aligns="center">
    <img src="https://physionet.org/files/eegmmidb/1.0.0/64_channel_sharbrough.png" height=400px>
</div>

Each subject performs 14 experimental runs, including periods of rest with open and closed eyes, as well as tasks involving the opening and closing of fists or feet in response to visual targets, both in actual execution and imagination.


### Data visualization



<!-- Citation -->
### Original publication
<a href="http://www.ncbi.nlm.nih.gov/pubmed/15188875">
Schalk G, McFarland DJ, Hinterberger T, Birbaumer N, Wolpaw JR. BCI2000: a general-purpose brain-computer interface (BCI) system. IEEE Trans Biomed Eng. 2004 Jun;51(6):1034-43. doi: 10.1109/TBME.2004.827072. PMID: 15188875.
</a>