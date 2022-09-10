import os
from glob import glob

ages = glob('expr/checkpoint/UTKFace_tau*_age_lambda_0.0_*')
new_name_template = lambda temp, seed: f'expr/checkpoint/UTKFace_tau{temp}_age_SimCLR_lambda_0.0_seed_{seed}'

tempdict = {
    '01': 0.1,
    '05': 0.5,
    '001': 0.01,
    '1': '1.',
    '5': '5.',
    '10': '10.',
    '15': '15.'
}

for ind, folder in enumerate(ages):
    old_temp = folder.split('/')[-1].split('UTKFace_tau')[1].split('_')[0]
    old_seed = folder.split('/')[-1].split('_')[-1]
    new_temp = tempdict[old_temp]
    new_name = new_name_template(new_temp, old_seed)
    os.rename(folder, new_name)

