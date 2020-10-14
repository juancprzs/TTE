# TAR
Instructions:
```bash
wget https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh
bash Anaconda3-2020.07-Linux-x86_64.sh
```

Here, you'll probably have to restart connection with the machine, then come back and run:
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

The spreadsheet is (here)[https://docs.google.com/spreadsheets/d/13iskg4cQlvAgvLB3HvPuq5tRlykhMZMDv_uHKHUxeZo/edit#gid=0].
