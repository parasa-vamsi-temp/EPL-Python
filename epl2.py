__author__ = "Srinivas Vamsi Parasa"
__contact__ = "srinivas.vamsi.parasa@intel.com"

import random as rnd
import os
import pickle
from operator import itemgetter
from copy import deepcopy
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

class Executor:

    def __init__(self, nsensors=72, fan_in=7, test_cycles=5):
        assert fan_in <= nsensors
        self.nsensors = nsensors
        self.mcs = None
        self.gc_stored_odors = None
        self.MC_TO_GC_FANIN = fan_in
        self.MAX_ITERS = test_cycles
        self.EMIN = 0.03

    @property
    def nodors(self):
        return len(self.gc_stored_odors)

    def train(self, train_data):
        self.gc_stored_odors = train_data

    def test(self, test_data):
        print("test", "*"*10)
        self.mcs = test_data
        self.mcs_new = deepcopy(self.mcs)
        self.odor_vote = [-1] * len(self.mcs)
        num_iters = 0
        energy = 1.0
        while energy > self.EMIN:
            for mc_id, mc in enumerate(self.mcs):
                best_match_odor_id = self.get_best_match()
                best_match_odor = self.gc_stored_odors[best_match_odor_id]
                if mc != best_match_odor[mc_id]:
                    self.mcs_new[mc_id] = best_match_odor[mc_id]
                self.odor_vote[mc_id] = best_match_odor_id
            self.mcs = self.mcs_new
            odor_predicted = self.highest_vote()
            energy = self.energy(odor_id=odor_predicted)
            num_iters += 1

        print("old odor     = {}".format(test_data))
        print("cleaned odor = {}".format(self.mcs))
        print("matched odor = {}".format(self.gc_stored_odors[odor_predicted]))
        print("odor vote    = {}".format(self.odor_vote))
        print("odor predicted={}; took num_iters={}".format(odor_predicted, num_iters))
        print("*"*10)
        return odor_predicted


    def get_best_match(self):
        best_match_odor_id = None
        best_match_count = -1
        for odor_id in range(len(self.gc_stored_odors)):
            receptive_field = rnd.sample(range(0, self.nsensors), self.MC_TO_GC_FANIN)
            # print("receptive_field = {}".format(receptive_field))
            match_count = self.num_matches(odor_id, receptive_field)
            if match_count > best_match_count:
                best_match_count = match_count
                best_match_odor_id = odor_id
        return best_match_odor_id

    
    def num_matches(self, odor_id, receptive_field):
        stored_odor = self.gc_stored_odors[odor_id]
        count = 0
        for i in receptive_field:
            if self.mcs[i] == stored_odor[i]:
                count += 1
        return count

    def highest_vote(self):
        votes = set(self.odor_vote)
        highest_vote = None
        highest_count = 0
        for vote in votes:
            count = self.odor_vote.count(vote)
            if count > highest_count:
                highest_count = count
                highest_vote = vote
        return highest_vote

    def energy(self, odor_id):
        num_matches = self.num_matches(odor_id=odor_id, receptive_field=range(len(self.mcs)))
        return (self.nsensors - num_matches)/self.nsensors

def gen_training_data(numOdors=1, numSensors=72, sparsity=0.5, MAX=20):
    """ generate synthetic training data for EPL network"""
    trainingData = []
    for i in range(0, numOdors):
        trainingData.append([])
        for j in range(0, numSensors):
            if rnd.random() > sparsity:
                trainingData[i].append(rnd.randint(0, MAX))
            else:
                trainingData[i].append(0)
    return trainingData

def gen_testing_data(trainingData, occlusionPercent, numTestSamples, MAX=20):
    """ generate synthetic testing data for EPL network"""
    data = trainingData
    p = occlusionPercent
    n = numTestSamples
    occludedData = []
    nsensors = len(data[0])

    for i in range(0, len(data)):
        ndim = len(data[i])  # dimension of data
        for j in range(0, n):
            occludedData.append([])
            affected_ids = rnd.sample(range(ndim), int(p * ndim))
            for k in range(0, ndim):
                if k in affected_ids:
                    occludedData[i * n + j].append(
                        rnd.randint(0, MAX))
                else:
                    occludedData[i * n + j].append(data[i][k])
    return occludedData

def plot_confusion_mat(actual, predicted, labels, doplt=True):
    cm = confusion_matrix(actual, predicted, labels)
    print(cm)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm)
    print('Diag of Confusion Matrix: ', np.diag(cm))
    avg_acc = 100 * np.average(np.diag(cm))
    print('Avg accuracy:{:.2f}%'.format(avg_acc))

    if not doplt:
        return avg_acc
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    # plt.title('EPL Confusion Matrix')
    fig.colorbar(cax)
    l = len(labels)
    ax.set_xticks(np.arange(l))
    ax.set_yticks(np.arange(l))
    # ax.set_xticklabels([''] + labels)
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right") #, rotation_mode="anchor")
    plt.xlabel('Predicted/Recalled Odor')
    plt.ylabel('Actual Odor')
    plt.show()

def test1(nsensors=72, nodors=2, ntest_samples=10, conn_prob=0.1, seed=0, occlusion_percent=0.8):
    fan_in = int(conn_prob * nsensors)
    ex = Executor(nsensors=nsensors, fan_in=fan_in, test_cycles=3)
    tr = gen_training_data(numOdors=nodors, numSensors=nsensors)
    ex.train(train_data=tr)
    print(ex.gc_stored_odors)
    print(ex.nodors)

    tst = gen_testing_data(trainingData=tr, occlusionPercent=occlusion_percent, numTestSamples=5)
    print(tst)
    for t in tst:
        ex.test(test_data=t)
    #ex.test(test_data=tst[0])

def test_fixed_occlusion_percent(nsensors=72, nodors=2, ntest_samples=10, fan_in=0.1,
          seed=0, occlusion_percent=0.8, doplt=True):
    if seed is not None:
        rnd.seed(seed)
    tr = gen_training_data(numSensors=nsensors, numOdors=nodors)
    tst = gen_testing_data(trainingData=tr,
                           occlusionPercent=occlusion_percent,
                           numTestSamples=ntest_samples)
    fan_in = int(fan_in * nsensors)
    ex = Executor(nsensors=nsensors, fan_in=fan_in, test_cycles=5)
    ex.train(train_data=tr)

    acc_map = {}
    labels = ['odor[{}]'.format(i) for i in range(nodors)]
    predicted = []
    actual = []
    for odor_id in range(nodors):
        count = 0
        beg = ntest_samples * odor_id
        print('*'*10)
        for td in tst[beg: beg + ntest_samples]:
            # td = tst[0]
            odor_predicted = ex.test(test_data=td)
            actual.append('odor[{}]'.format(odor_id))
            predicted.append('odor[{}]'.format(odor_predicted))
            print('odor_id[{}] predicted={}'.format(odor_id, odor_predicted))
            if odor_id == odor_predicted:
                count += 1
        acc_map[odor_id] = int(100 * count/ntest_samples)
    print(acc_map)
    print(actual)
    print(predicted)
    avg_acc = plot_confusion_mat(actual=actual, predicted=predicted, labels=labels, doplt=doplt)
    return avg_acc


def plot_occlusion_vs_accuracy(nodors=10, num_occlusion_points=5):
    occlusion_percents = np.linspace(0, 1, num_occlusion_points)
    avg_acc_all = []
    for occlusion_percent in occlusion_percents:
        avg_acc = test_fixed_occlusion_percent(nsensors=72, nodors=nodors, ntest_samples=50,
        occlusion_percent=occlusion_percent, fan_in=0.2, seed=100, doplt=False)
        avg_acc_all.append(avg_acc)
    
    plt.figure()
    plt.plot(occlusion_percents, avg_acc_all, linestyle='dashed', marker='o', markerfacecolor='blue')
    plt.title("General EPL - Occlusion vs Accuracy")
    plt.xlabel("Proportion occluded")
    plt.ylabel("% Correct")
    plt.show()

        
if __name__ == "__main__":
    # test_fixed_occlusion_percent(nsensors=72, nodors=10, ntest_samples=5, occlusion_percent=0.5, fan_in=0.2, seed=100)

    # Below plots the accuracy by varying the occlusion percentage from 0 to 100 ([0, 1.0])
    # plot_occlusion_vs_accuracy(nodors=5, num_occlusion_points=5)

    # This takes a LOT of time but plots a nice curve which matches with Nature paper
    plot_occlusion_vs_accuracy(nodors=10, num_occlusion_points=10)

        

