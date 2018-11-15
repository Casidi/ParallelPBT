from toy_model import ToyModel
from constants import generate_random_hparam
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class SequentialPBT:
    def __init__(self, pop_size, epochs_per_round, 
                    do_exploit=True, do_explore=True):
        print("Initializing populations")
        self.all_models = []
        self.epochs_per_round = epochs_per_round
        self.do_exploit = do_exploit
        self.do_explore = do_explore

        for i in range(pop_size):
            print("Generating HPs for model {}".format(i))
            hp = generate_random_hparam()
            self.all_models.append(ToyModel(i, hp, 'savedata'))

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

    def explore(self):
        print("Exploring...")

def main():
    if os.path.isdir('savedata') == False:
        os.mkdir('savedata')

    test = SequentialPBT(10, 4)
    test.train(3)

if __name__ == "__main__":
    main()
