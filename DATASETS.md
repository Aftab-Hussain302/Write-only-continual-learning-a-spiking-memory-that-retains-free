# Datasets

Datasets are not bundled. Put them under one root (`--data_root`, e.g. `~/data`) in this layout:

| dataset | folder under the data root | source |
|---|---|---|
| MNIST | `MNIST/raw/` (the four idx files, decompressed) | http://yann.lecun.com/exdb/mnist/ (torchvision downloads it if absent) |
| CIFAR-10 | `cifar-10-batches-py/` | `cifar-10-python.tar.gz` from https://www.cs.toronto.edu/~kriz/cifar.html (torchvision downloads it if absent) |
| CIFAR-100 | `cifar-100-python/` | `cifar-100-python.tar.gz` from https://www.cs.toronto.edu/~kriz/cifar.html (torchvision downloads it if absent) |
| N-MNIST | `nmnist/Train/<digit>/*.bin` (and `nmnist/Test/...`, unused) | Orchard et al. 2015, the official `Train.zip` from https://www.garrickorchard.com/datasets/n-mnist |
| SHD | `shd/shd_train.h5` | Cramer et al. 2020, `shd_train.h5` from https://zenkelab.org/datasets/ |
| SSC (optional) | `ssc/ssc_train.h5` | Cramer et al. 2020, `ssc_train.h5` from https://zenkelab.org/datasets/ |
| Tiny-ImageNet | `tiny-imagenet-200/wnids.txt`, `tiny-imagenet-200/train/<wnid>/images/*.JPEG` | the official `tiny-imagenet-200.zip`, http://cs231n.stanford.edu/tiny-imagenet-200.zip |

`python3 scripts/prepare_data.py --data_root <root>` loads every dataset once through the real loaders, reports what it found,
and builds the frozen front-end feature caches (below) before the parallel campaign starts. A dataset that cannot be loaded is
reported as an error; nothing is ever substituted for it.

N-MNIST `.bin` files and the SHD and SSC `.h5` files are parsed by the loaders directly (no event-data library; `h5py` is needed).
N-MNIST events are binned into T = 6 frames of 2 x 34 x 34; SHD and SSC spike trains into T = 20 frames of 700 channels over the
recording window (1.4 s for SHD, 1.0 s for SSC), counts clipped to one. Only the samples the protocol needs are materialised (train
and test per class, see `rewire/configure.py`). SSC has 35 classes: 28 form the stream and 7 are held out.

CIFAR-10, CIFAR-100 and Tiny-ImageNet are presented to every method through the same frozen ImageNet-pretrained ResNet-18
(torchvision `IMAGENET1K_V1`, downloaded on first use). Its 512-d features are extracted once per dataset and cached as
`<root>/_features_resnet18_<dataset>_<N>.npz`; feature standardisation uses a fixed 10 % subset of the images, which is then
removed from every stream and test set. For the domain-shift stage the features of the rotated images are extracted once per seed
(the rotation's sign follows the seed) and cached next to them. Tiny-ImageNet is read as 64 x 64 uint8 images (1.2 GB for 100k
images) and scaled per batch inside the ResNet pass. Tiny-ImageNet overlaps the front end's pretraining data; the paper says so and
anchors the 80-class claims on CIFAR-100.
