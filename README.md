# TAR
# # Install conda
```bash
wget https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh
bash Anaconda3-2020.07-Linux-x86_64.sh
```

# # Large files
We'll be dealing with large files. It will be easier for us to manage it directly through GitHub. There's a tool called ["Git Large File Storage"](https://git-lfs.github.com/). We're still learning to use it, so there could be a couple problems.

In any case, we need to install it. First, ask conda to install it:
```bash
conda install -c conda-forge git-lfs
```

Then set it up by running
```bash
git lfs install
```

# # The repo and the environment
Clone the repo and create the environment
```bash
git clone https://github.com/juancprzs/TAR.git
cd TAR
conda env create -f utils/upd_pt.yml
```

Activate the `upd_pt` environment by running
```bash
conda activate upd_pt
```

Also, install AutoAttack:
```bash
pip install git+https://github.com/fra31/auto-attack
```
And install `tqdm`
```bash
pip install tqdm
```

# # Results
The spreadsheet is [here](https://docs.google.com/spreadsheets/d/13iskg4cQlvAgvLB3HvPuq5tRlykhMZMDv_uHKHUxeZo/edit#gid=0).
