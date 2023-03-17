PUBLIC_HYPERPARAMS={
    "epochs":8,
    "train_batch":64,
    "lr":0.001
}

PRIVATE_HYPERPARAMS={
    "epochs":100,
    "train_batch":64,
    "lr":0.0001
}

TEST_BATCH=1024

PRUNE_TYPE="magnitude"
PRUNE_AMOUNT=0.7
PRUNE_LARGEST=True

GN_GROUPS=4

NOISE_MULTIPLIER=0.75
MAX_GRAD_NORM=1
DELTA=1e-5

NUM_WORKERS=0
PIN_MEMORY=True

LOC="./experiment/"

CIFAR10_MEAN=(0.5,0.5,0.5)
CIFAR10_STD=(0.25,0.25,0.25)

NUM_CLASSES=10

if __name__ == "__main__":
    pass