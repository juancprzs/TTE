# TTE
This is the official implementation of the paper *Enhancing Adversarial Robustness via Test-time Transformation Ensembling* (available [here](https://openaccess.thecvf.com/content/ICCV2021W/AROW/papers/Perez_Enhancing_Adversarial_Robustness_via_Test-Time_Transformation_Ensembling_ICCVW_2021_paper.pdf)), published at the ICCV 2021 workshop on [Adversarial Robustness in the Real World](https://iccv21-adv-workshop.github.io).
## Install conda
```bash
wget https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh
bash Anaconda3-2020.07-Linux-x86_64.sh
```

## The repo and the environment
Clone the repo and create the environment
```bash
git clone https://github.com/juancprzs/TTE.git
cd TTE
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

## Usage example
This code internally manages the computation of adversaries by partitioning the dataset into chunks and evaluating each chunk. This dynamic will allow us to parallelize adversary computation across jobs. However, the code is also capable of conducting the computation of adversaries for the entire dataset. Here I'll demonstrate how the _same_ results can be achieved both by doing the full thing _vs._ the "chunked" version of the computation.

We'll check with the `--cheap` flag so that everything runs in reasonable time. I'll run things on top of the standard version of TRADES. 

### Full version
Run
```bash
python main.py --checkpoint check1 --cheap
```
This will save the output of the process in the `check1` directory. This directory has four items:
* `advs`: a directory with several files of the form `advs_chunk2of10_1000to2000.pth`. The meaning of the naming of this file is the following: this is the result of evaluating the chunk number 2 out of 10 (total) chunks. This chunk corresponds to the instances you'd get by querying the dataset like this `dataset[1000:2000]`. Each of the `.pth` files is a dictionary with two keys: 'advs' and 'labels'. The item that corresponds to the 'advs' key is, in turn, a dictionary with as many entries as attacks were run. Each of the entries is a tensor in which the adversaries are stored, _i.e._ `torch.load('advs_chunk2of10_1000to2000.pth')['advs']['square'].shape` would give you, in this case `[1000, 3, 32, 32]`. The item that corresponds to the 'labels' key is simply the labels that correspond to each of the images.
* `logs`: a directory with several files of the form `results_chunk2of10_1000to2000.txt`. The meaning of the naming of these files is analogous to those inside `advs`. The file reports the accuracies under the attacks, the clean accuracy, and the number of instances that correspond to that evaluation. For instance, if you run `cat check1/logs/results_chunk4of10_3000to4000.txt`, you should get:
```bash
apgd-ce:59.70
square:81.90
rob acc:59.70
clean:84.30
n_instances:1000
```
* `info_chunk_all.txt`: a text file with the parameters with which this experiment was run. The `all` in the file's name refers to the fact that this experiment was _not_ run in chunks.
* `results.txt`: a text file with the accuracy results of the run. **This is the file we care about!** Its contents are analogous to those of the text files under the `logs` dir. The only difference is that these are the results considering _all_ the chunks of data, instead of a particular one. For this experiment, you should get
```bash
apgd-ce:58.84
square:81.51
rob acc:58.84
clean:84.92
n_instances:10000
```

### Chunked version
To simulate the chunked version, run
```bash
for i in {1..10}; do 
    python main.py --checkpoint check2 --cheap --num-chunk $i; 
done
```
Besides two minor differences, running these lines will produce the _same_ results as the previous section. The two differences are the following:
* There will not only one file with the parameters, there will be several files of the form `info_chunk_X.txt` (instead of `info_chunk_all.txt`). This is because, technically, it wasn't a single run of the `main.py` script. Of course, _all_ these files should have _exactly_ the same contents.
* There is no `results.txt` file. We have to generate it based on all the logs at the `logs` directory.

We get the final results by running
```bash
python main.py --checkpoint check2 --eval-files
```
This line will compute the final results based on the logs of the form `check2/logs/results_chunk*of*_*to*.txt`. The results will be saved, as expected, at `check2/results.txt`. You should get
```bash
apgd-ce:58.84
square:81.51
rob acc:58.84
clean:84.92
n_instances:10000
```
Which is the same one would obtain by following the instructions from the previous section.


## Citing our work
If you find our work useful, please consider citing us. The corresponding BibTex entry is:
```
@inproceedings{perez2021enhancing,
  title={Enhancing Adversarial Robustness via Test-time Transformation Ensembling},
  author={P{\'e}rez, Juan C and Alfarra, Motasem and Jeanneret, Guillaume and Rueda, Laura and Thabet, Ali and Ghanem, Bernard and Arbel{\'a}ez, Pablo},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2021}
}
```
