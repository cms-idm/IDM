> **Run 3 setup:** This README covers the legacy Run-2 / `cmslpc-sl7` / conda setup and is
> out of date. For the current Run 3 (electron + muon) analysis environment and how to get
> things running, see [`../SETUP_Run3.md`](../SETUP_Run3.md).

# Coffea analysis tools for iDMe

This directory contains all the python-based analysis tools for iDMe. The analysis pipeline (i.e. applying cuts, filling histograms, etc.) is built using the [coffea](https://coffeateam.github.io/coffea/) framework for columnar analysis. Here's a summary of everything in this area:
- `analysisTools` : core functionality for the `coffea`-based analysis, including code for the `Analyzer` and `Processor` objects.
- `configs` : houses three different kinds of config files we use with our coffea processors: histograms, cuts, and samples. The dedicated README inside this directory explains things in more detail
- `condor` : tools for submitting `coffea` analysis jobs to condor. See README inside for details
- `studies` : houses directories and notebooks for various studies. Feel free to add your own stuff here!
- `exampleNotebooks` :  some jupyter notebooks with simple coffea/analysis examples

# Setting up jupyter on the LPC
If you haven't already, set up `miniconda` or `miniforge` in your `/nobackup` area on LPC so that you can use and create conda environments. To get a working `coffea` setup, you can use the `coffea_env.yml` file in this directory to create a new conda environment:
```bash
conda create -f coffea_env.yml
```
In your `base` conda environment, install `jupyterlab` and `nb_conda_kernels`. To run a jupyterlab session on the LPC and interact with it via your local browser, run the following commands from your terminal:
```bash
ssh -Y -L 8888:localhost:8888 YOUR_USERNAME@cmslpc-sl7.fnal.gov
cd /path/to/your/iDMe/repo/clone
cd python_analysis
jupyter lab --no-browser --port=8888
```
In the terminal output, you should see a link that you can paste into your browser. This will open up an interactive jupyter session where you can create and run notebooks. The `nb_conda_kernels` package should make the `coffea` environment available to select as a notebook kernel.