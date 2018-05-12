from hashing import main
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import pickle
import time

def data_distribution():

    for x in ('0.01','0.001','0.0001'):
        target_hash = 1
        true_hash = 2

        all_data = []
        args = ['./test_files/target_random', '--source', 'start', '--output', 'out', '--iters', '500','--reg-coeff',x]

        while len(all_data)<409600:
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

        pickle.dump(all_data,open('datadistribution'+x[2:],'wb'))
'''
    fig, ax = plt.subplots()

    # the histogram of the data
    n, bins, patches = ax.hist(all_data, bins=np.arange(min(all_data), max(all_data) + 0.005, 0.005))

    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    plt.show()
'''    

#[4, 13, 20, 20] [1.593271255493164, 2.5054832055018497, 7.060126960277557, 9.448571419715881] 128,256,512,1024
def performance():
    successfulrate =[]
    averagetime = []

    target_hash = 1
    true_hash = 2

    all_data = []
    args = ['./test_files/target_random', '--source', 'start', '--output', 'out', '--iters', '300']

    for j in (128,256,512,1024):
        success = 0
        totaltime = 0
        for i in range(20):
            print(i)
            start_file = open('start','wb')
            start_file.write(os.urandom(j))
            start_file.close()
            starttime = time.time()
            start_hash,target_hash,data = main(args)
            stoptime = time.time()
            if data is not None:
                success+=1
                totaltime += stoptime-starttime
                print(stoptime-starttime)
        successfulrate.append(success)
        averagetime.append(totaltime/success)
    
    print(successfulrate,averagetime)

#[17, 17, 20, 20, 18] [5.191271781921387, 3.8180168376249424, 4.426693153381348, 4.4814985871315, 5.4564668734868365]
def performance_lr():
    successfulrate =[]
    averagetime = []

    target_hash = 1
    true_hash = 2

    all_data = []
    args = ['./test_files/target_random', '--source', 'start', '--output', 'out', '--iters', '300','--adam-lr']

    for j in ('0.1','0.5','1.0','1.5','2'):
        success = 0
        totaltime = 0
        for i in range(20):
            print(i)
            start_file = open('start','wb')
            start_file.write(os.urandom(512))
            start_file.close()
            starttime = time.time()
            start_hash,target_hash,data = main(args[:]+[j])
            stoptime = time.time()
            if data is not None:
                success+=1
                totaltime += stoptime-starttime
                print(stoptime-starttime)
        successfulrate.append(success)
        averagetime.append(totaltime/success)
    
    print(successfulrate,averagetime)

#[0, 0, 0, 15, 20, 20, 20] [0, 0, 0, 5.782202339172363, 5.980342841148376, 5.684534215927124, 5.559066712856293]
def performance_reg():
    successfulrate =[]
    averagetime = []

    target_hash = 1
    true_hash = 2

    all_data = []
    args = ['./test_files/target_random', '--source', 'start', '--output', 'out', '--iters', '300','--reg-coeff']

    for j in ('1','0.5','0.1','0.01','0.001','0.0001','0.00001'):
        success = 0
        totaltime = 0
        for i in range(20):
            print(i)
            start_file = open('start','wb')
            start_file.write(os.urandom(512))
            start_file.close()
            starttime = time.time()
            start_hash,target_hash,data = main(args[:]+[j])
            stoptime = time.time()
            if data is not None:
                success+=1
                totaltime += stoptime-starttime
                print(stoptime-starttime)
        successfulrate.append(success)
        if success:
            averagetime.append(totaltime/success)
        else:
            averagetime.append(0)
    
    print(successfulrate,averagetime)

def numpy():
    target_hash = 1
    true_hash = 2

    all_data = []
    args = ['./test_files/target_random', '--source', 'start', '--output', 'out', '--iters', '500','--reg-coeff','0.0001']

    start_file = open('start','wb')
    start_file.write(os.urandom(512))
    start_file.close()
    start_hash,target_hash,data = main(args)

    if data is not None:
        low, high = data[data < .5], data[data >= .5]
        print('Low mean: {:.4f}, std: {:.4f}, min: {:.4f}, max: {:.4f}'.format(
            np.mean(low), np.std(low), np.min(low), np.max(low)))
        print('High mean: {:.4f}, std: {:.4f}, min: {:.4f}, max: {:.4f}'.format(
            np.mean(high), np.std(high), np.min(high), np.max(high)))

        print(low.size)
        lowl = list(low.flat)
        print('low mean:' + str(min(lowl))+ str(max(lowl))+str(np.mean(lowl)))

        print(high.size)
        highl = list(high.flat)


        true_hash,_,_ = main(['out'])
        print(target_hash)
        print(true_hash)
        all_data.extend(list(data.flat))

def successfulrate_time_plot():

    '''
    # Create some mock data
    t = [128,256,512,1024]
    data1 = [4, 13, 20, 20]
    data1 = [x/20 for x in data1]
    data2 = [1.593271255493164, 2.5054832055018497, 7.060126960277557, 9.448571419715881]
    '''
    '''
    #[17, 17, 20, 20, 18] [5.191271781921387, 3.8180168376249424, 4.426693153381348, 4.4814985871315, 5.4564668734868365]
    t= [0.1,0.5,1.0,1.5,2]
    data1 = [17, 17, 20, 20, 18]
    data1 = [x/20 for x in data1]
    data2 = [5.191271781921387, 3.8180168376249424, 4.426693153381348, 4.4814985871315, 5.4564668734868365]
    '''
    #[0, 0, 0, 15, 20, 20, 20] [0, 0, 0, 5.782202339172363, 5.980342841148376, 5.684534215927124, 5.559066712856293]
    t = [0.1,0.01,0.001,0.0001,0.00001]
    t = np.log10(t)
    data1=[0, 15, 20, 20, 20]
    data1 = [x/20 for x in data1]
    data2=[0, 5.782202339172363, 5.980342841148376, 5.684534215927124, 5.559066712856293]
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('regularization coefficient (log(coefficient))')
    ax1.set_ylabel('success rate (%)', color=color)
    ax1.plot(t, data1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('average time (s)', color=color)  # we already handled the x-label with ax1
    ax2.plot(t, data2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

#datadistribution
def plot():
    '''
    for x in ('01','001','0001'):
        print(x + '-----------------------')
        data = pickle.load(open('datadistribution'+x,'rb'))
        low = [x for x in data if x < 0.5]
        high = [x for x in data if x >= 0.5]
        print('Low mean: {:.4f}, std: {:.4f}, min: {:.4f}, max: {:.4f}'.format(
                np.mean(low), np.std(low), np.min(low), np.max(low)))
        print('High mean: {:.4f}, std: {:.4f}, min: {:.4f}, max: {:.4f}'.format(
                np.mean(high), np.std(high), np.min(high), np.max(high)))
    '''

    data1 = pickle.load(open('datadistribution01','rb'))
    data2 = pickle.load(open('datadistribution001','rb'))
    data3 = pickle.load(open('datadistribution0001','rb'))

    fig, ax = plt.subplots()

    # the histogram of the data
    #n, bins, patches = ax.hist([data1,data2,data3],label=['01', '001','0001'],bins=np.arange(-0.1,1.1, 0.005))
    
    ax.hist(data1,color='r',alpha=0.5,label = '0.01', bins=np.arange(-0.1,1.1, 0.005))
    ax.hist(data2,color='g',alpha=0.5,label = '0.001', bins=np.arange(-0.1,1.1, 0.005))
    ax.hist(data3,color='b',alpha=0.5,label = '0.0001', bins=np.arange(-0.1,1.1, 0.005))
    ax.legend(loc='upper left') 
    ax.set_xlabel('Pseudo-collision values')
    ax.set_ylabel('frequency')
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
    #numpy()
    plot()
    #data_distribution()
    #dispatch(sys.argv[1:])
    #performance_reg()
    #successfulrate_time_plot()


