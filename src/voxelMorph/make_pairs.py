import csv
import itertools
from tqdm import tqdm

def gen_pairwise_permutations(triplet):
    return list(itertools.permutations(triplet, 2))
def gen_pairwise_combinations(triplet):
    return list(itertools.combinations(triplet, 2))

input_csv = '../../dev_dataset/triples.csv'
perm_csv = '../../dev_dataset/pairwise_permutations.csv'
comb_csv = '../../dev_dataset/pairwise_combinations.csv'

with open(input_csv, 'r') as infile:
    total_rows = sum(1 for _ in infile)

with open(input_csv, 'r') as csvfile, open(comb_csv, 'w', newline='') as comb_file, open(perm_csv, 'w', newline='') as perm_file:
    reader = csv.reader(csvfile)
    writer_comb = csv.writer(comb_file)
    writer_perm = csv.writer(perm_file)

    for row in tqdm(reader, total=total_rows, desc="Processing csv"):
        triplet = [item for item in row if item]
        pairwise_combs = gen_pairwise_combinations(triplet)
        pairwise_perms = gen_pairwise_permutations(triplet)
        writer_comb.writerows(pairwise_combs)
        writer_perm.writerows(pairwise_perms)
