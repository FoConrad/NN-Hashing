from hashing import main
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import pickle

def data_distribution():
    target_hash = 1
    true_hash = 2

    all_data = []
    args = ['./test_files/target_random', '--source', 'start', '--output', 'out', '--iters', '500','--reg-coeff','0.0001']

    while len(all_data)<100000:
        print(len(all_data))
        start_file = open('start','wb')
        start_file.write(os.urandom(512))
        start_file.close()
        start_hash,target_hash,data = main(args)
        if data is not None:
            true_hash,_,_ = main(['out'])
            print(target_hash)
            print(true_hash)
            all_data.extend(list(data.flat))

    pickle.dump(all_data,open('datadistribution0001','wb'))

    fig, ax = plt.subplots()

    # the histogram of the data
    n, bins, patches = ax.hist(all_data, bins=np.arange(min(all_data), max(all_data) + 0.005, 0.005))

    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    plt.show()

def plot():
    data = pickle.load(open('datadistribution','rb'))

    fig, ax = plt.subplots()
    
    # the histogram of the data
    n, bins, patches = ax.hist(data, bins=np.arange(0.99, 1.01, 0.001))

    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    plt.show()


EXPERIMENTS_MAP = {
    'data_distribution':data_distribution
}



def dispatch(experiments):
    
    for e in experiments:
        EXPERIMENTS_MAP[e]()

if __name__ == '__main__':
    #plot()
    dispatch(sys.argv[1:])


