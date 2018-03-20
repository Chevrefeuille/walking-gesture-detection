import json
from pprint import pprint

if __name__ == "__main__":
    data = json.load(open('../../st-gcn/data/kinetics-skeleton/kinetics_train/kwc5Xz8Kngw.json'))
    pprint(data)