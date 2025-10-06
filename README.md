# FPGA-EEG with FINN

This repository demonstrates how to train quantized version of the Deep4 EEG model from [Braindecode](https://braindecode.org) and deploy it
onto FPGA hardware using [FINN](https://finn.readthedocs.io/en/latest/).

## Prerequisites

Before starting, make sure you have:
- **Vivado** 2022.2 installed
- **Docker** installed

Set the following environment variables in your shell/profile:

- `FINN_XILINX_PATH=/your/path/to/Xilinx`
- `FINN_XILINX_VERSION=2022.2`

On my system the Xilinx directory was found at `/tools/Xilinx`. This project has only been tested with the 2022.2 version of **Vivado**.

## Setup

Clone the repo and pull submodules:
```bash   
git clone https://github.com/danielflood/fpga-eeg.git
cd fpga-eeg
git submodule update --init --recursive
``` 
## Running the Project

Start the **FINN Docker** environment with:

```bash
 ./run-docker.sh notebook
 ``` 


This will build the **Docker** image (if not already built) and launch **Jupyter Lab** at the project root.

From there, you can open the notebooks folder and follow the flow:

1.  **Train** the quantized Deep4 model (on fake EEG data for now).
2.  **Synthesize** the trained model into FPGA hardware with PYNQ drivers.

## Repository Layout

- `docker/Dockerfile.project` - Dockerfile that wraps the base image and installs requirements.txt
- `lib/finn/` – FINN submodule (pinned to a specific commit).
- `notebooks/` – Jupyter notebooks for training and synthesis.
- `run-docker.sh` – Script to launch the Dockerized FINN environment.
- `src/` – Supporting code (i.e. quantized model)

## Notes

- Hardware builds may take significant time.
- You’ll need a supported FPGA development board (e.g. PYNQ-Z2) for deployment.

## Known Issues

- This project uses a very old version of **Braindecode**. You will likely get many deprecation warnings.
- In the past I've gotten a **Vivado** stack overflow error when running synthesis. I got around this by changing **line 260** in `./lib/finn/src/finn/transformation/fpgadataflow/make_zynq_proj.py` to:
  ```python
  f.write("vivado -stack 10000 -mode batch -source %s\n" % ipcfg)
  ``` 

## References
1. Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J.,      Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F. & Ball, T. (2017).   
  *Deep learning with convolutional neural networks for EEG decoding and visualization.*  
  Human Brain Mapping, Aug. 2017. [https://doi.org/10.1002/hbm.23730](https://doi.org/10.1002/hbm.23730)
2. Lee, M.-H., Kwon, O.-Y., Kim, Y.-J., Kim, H.-K., Lee, Y.-E., Williamson, J., Fazli, S., & Lee, S.-W. (2019).  
  *Supporting data for "EEG Dataset and OpenBMI Toolbox for Three BCI Paradigms: An Investigation into BCI Illiteracy"*  
  [Data set]. GigaScience Database. https://doi.org/10.5524/100542
3. K. Zhang, N. Robinson, S.-W. Lee, C. Guan (2021)  
   *Adaptive transfer learning for EEG motor imagery classification with deep Convolutional Neural Network.*  
   Neural Networks. https://doi.org/10.1016/j.neunet.2020.12.013 

## Licence
MIT License — see [LICENSE](./LICENCE) for details.