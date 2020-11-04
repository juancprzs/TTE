import os.path as osp
from glob import glob
import matplotlib.pyplot as plt
from argparse import ArgumentParser

parser = ArgumentParser(description='Generate gradient obfuscation plots')
parser.add_argument('--exp', type=str, default='eps', 
                    help='which ablation to plot', choices=['eps','n_iters'])
args = parser.parse_args()

upper_lim = 65 if args.exp == 'eps' else 101
root = f'obfuscation_ablations/{args.exp}'

def readlines_and_assert(filee):
    with open(filee, 'r') as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines] 
    lines = { x.split(':')[0] : float(x.split(':')[1]) for x in lines }
    assert int(lines['n_instances']) == 10_000 # for CIFARs!
    
    return lines['rob acc'], lines['clean']

for data in ['Baseline', 'TTE']:
    clean_acc = None
    x_axis = [0, ]
    rob_accs = []
    for val in range(1, upper_lim):
        if args.exp == 'eps':
            if data == 'Baseline':
                filee = f'{root}/{data}/obf_abl_{val}of255/results.txt'
            else:
                filee = f'{root}/{data}/obf_abl_{val}of255_fc4/results.txt'
        else:
            if data == 'Baseline':
                filee = f'{root}/{data}/obf_abl_n_iter{val}/results.txt'
            else:
                filee = f'{root}/{data}/obf_abl_n_iter{val}_fc4/results.txt'
        
        try:
            rob_acc, clean = readlines_and_assert(filee)
            # assertion on clean acc
            if clean_acc is None:
                clean_acc = clean
            else:
                assert clean_acc == clean
            rob_accs.append(rob_acc)
            x_axis.append(val)
        except:
            pass

    rob_accs.insert(0, clean_acc) # insert clean acc

    plt.plot(x_axis, rob_accs, label=data)

plt.grid(True)
plt.legend()
element = r'attack strength ($\epsilon$)' if args.exp == 'eps' else 'iterations'
# Labels
plt.title(f'Robust accuracy vs. {element}')
plt.ylabel('Robust accuracy')
plt.xlabel(element.capitalize())
# Limits
plt.ylim(0, clean_acc + 5.)
plt.xlim(0, upper_lim + 1)

plt.tight_layout()

figname = f'{args.exp}.png'
plt.savefig(figname, dpi=200)
print(f'Saved figure to "{figname}"')
