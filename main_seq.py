from toy_model import ToyModel
from constants import generate_random_hparam

class SequentialPBT:
    def __init__(self, pop_size, epochs_per_round, 
                    do_exploit=True, do_explore=True):
        print("Initializing populations")
        self.all_models = []
        for i in range(pop_size):
            hp = generate_random_hparam()
            self.all_models.append(ToyModel(i, hp, 'savedata'))
        for m in self.all_models:
            m.train(1, 1)
        print(self.all_models)

def main():
    print("main")
    test = SequentialPBT(10, 4)

if __name__ == "__main__":
    main()
