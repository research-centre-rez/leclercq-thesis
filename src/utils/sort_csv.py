import csv
from operator import itemgetter

def sort_csv(csv_name):
    with open(csv_name) as f:
        data = list(csv.reader(f))

    data.sort(key=itemgetter(0))

    with open(csv_name, 'w') as f:
        csv.writer(f).writerows(data)
