from toy_model import ToyModel
from mnist_model import MNISTModel
from constants import generate_random_hparam
import os
import shutil
import math
import subprocess

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class SequentialPBT:
    def __init__(self, pop_size, epochs_per_round, 
                    do_exploit=True, do_explore=True):
        print("Initializing populations")
        self.all_models = []
        self.epochs_per_round = epochs_per_round
        self.do_exploit = do_exploit
        self.do_explore = do_explore
        self.pop_size = pop_size

        for i in range(pop_size):
            print("Generating HPs for model {}".format(i))
            hp = generate_random_hparam()
            #self.all_models.append(ToyModel(i, hp, 'savedata/model_'))
            self.all_models.append(MNISTModel(i, hp, 'savedata/model_'))

    def train(self, rounds_to_train):
        for i in range(rounds_to_train):
            print("Round {} starts".format(i))
            for m in self.all_models:
                print("Training model {}".format(m.cluster_id))
                m.train(self.epochs_per_round, 
                        rounds_to_train * self.epochs_per_round)

            if self.do_exploit:
                self.exploit()
            if self.do_explore:
                self.explore()
            print()

    def exploit(self):
        print("Exploiting...")
        all_values = []
        for i in self.all_models:
            all_values.append(i.get_values())
        
        all_values = sorted(all_values, key=lambda value: value[1])
        self.pop_size = len(all_values)
        '''print 'The ranking before exploit'
        for i in all_values:
            print 'graph {}, loss={}'.format(i[0], i[1])'''
        num_graphs_to_copy = math.ceil(self.pop_size / 4.0)
        for i in range(num_graphs_to_copy):
            bottom_index = i
            top_index = len(all_values) - num_graphs_to_copy + i
            all_values[bottom_index][1] = all_values[top_index][1]  # copy accuracy, not necessary
            all_values[bottom_index][2] = all_values[top_index][2]  # copy hparams

            source_dir = './savedata/model_' + str(all_values[top_index][0])
            destination_dir = './savedata/model_' + str(all_values[bottom_index][0])
            self.copyfiles(source_dir, destination_dir)

            self.all_models[all_values[bottom_index][0]].set_values(all_values[top_index])
            print('Copied: {} -> {}'.format(all_values[top_index][0], all_values[bottom_index][0]))

    def copyfiles(self, src_dir, dest_dir):
        if src_dir == dest_dir:
            print('Warning, src_dir and dest_dir are the same')
            return
        for i in os.listdir(dest_dir):
            path = os.path.join(dest_dir, i)
            if not os.path.isdir(path) and i != 'learning_curve.csv' and i != 'theta.csv' and not i.startswith('events.out') and not i.startswith('.nfs'):
                #print('Removing: {}'.format(path))
                subprocess.call(['rm', '-f', path])
        for i in os.listdir(src_dir):
            path = os.path.join(src_dir, i)
            if not os.path.isdir(path)  and i != 'theta.csv' and i != 'learning_curve.csv' and not i.startswith('events.out') and not i.startswith('.nfs'):
                #print('Copying: {}'.format(path))
                subprocess.call(['cp', path, dest_dir])

    def explore(self):
        print("Exploring...")
        for i in self.all_models:
            i.perturb_hparams()

def main():
    if os.path.isdir('savedata'):
        shutil.rmtree('savedata')
    os.mkdir('savedata')
    if os.path.isdir('datasets') == False:
        os.mkdir('datasets')

    test = SequentialPBT(4, 4)
    test.train(3)

if __name__ == "__main__":
    main()
