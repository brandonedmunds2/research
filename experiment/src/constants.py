EPOCHS=10
TRAIN_BATCH=64
LR=0.01
TEST_BATCH=1024
PHYSICAL_BATCH=64

PRUNE_TYPE="magnitude"
PRUNE_LAYERS=('fc1',)
PRUNE_AMOUNT=0.99
PRUNE_LARGEST=False

GN_GROUPS=4

NOISE_MULTIPLIER=1.0
MAX_GRAD_NORM=1.0
DELTA=1e-5

NUM_WORKERS=0
PIN_MEMORY=True

LOC="./experiment/"

CIFAR10_MEAN=(0.5,0.5,0.5)
CIFAR10_STD=(0.25,0.25,0.25)

NUM_CLASSES=10

def print_constants():
    print(PRUNE_TYPE)
    print(PRUNE_AMOUNT)
    print(PRUNE_LAYERS)
    print(PRUNE_LARGEST)

if __name__ == "__main__":
    pass