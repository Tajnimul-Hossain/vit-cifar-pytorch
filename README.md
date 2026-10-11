# Vision Transformer on CIFAR-10 and CIFAR-100

A Vision Transformer I wrote from scratch in PyTorch and trained on CIFAR-10 and CIFAR-100, with no pretrained weights. It reaches 92.9% test accuracy on CIFAR-10 and 74.7% on CIFAR-100 after 200 epochs. This started as a course assignment at North South University.

## Results

| Dataset | Final test accuracy | Best test accuracy | Final train accuracy |
|---|---|---|---|
| CIFAR-10 | 92.9% | 93.1% | 69.1% |
| CIFAR-100 | 74.7% | 74.8% | 59.9% |

Training curves are in `results/`.

Train accuracy sits well below test accuracy. That is expected here: training images go through CutMix, MixUp and label smoothing, which make the training task harder than the clean test set.

## Model

- Convolutional stem: two 3x3 stride-2 convolutions turn the 32x32 image into an 8x8 grid of tokens
- 12 transformer layers, 512-dim embeddings, 8 attention heads, MLP ratio 4
- LayerScale and stochastic depth (drop path 0.1) in every block
- Learnable position embeddings and a class token

## Training

- AdamW, learning rate 1e-3, weight decay 0.05, batch size 128
- 10 warmup epochs, then cosine decay to 1e-6, 200 epochs total
- Augmentation: random crop, AutoAugment, CutMix and MixUp
- Label smoothing 0.1, mixed precision
- Trained on one NVIDIA A100 in Google Colab

## Hyperparameter comparison

Four setups trained for 50 epochs each on CIFAR-10:

| Setup | Embed dim | Layers | Heads | Learning rate | Batch | Best test accuracy |
|---|---|---|---|---|---|---|
| Baseline | 384 | 8 | 6 | 1e-3 | 128 | 81.4% |
| Deep model | 512 | 12 | 8 | 1e-3 | 128 | 81.3% |
| Large batch | 512 | 12 | 8 | 2e-3 | 256 | 42.1% |
| High learning rate | 512 | 12 | 8 | 3e-3 | 128 | 26.0% |

At 50 epochs the deeper model gave no gain over the baseline. Raising the learning rate to 3e-3 made training diverge: the loss went to NaN and accuracy dropped to chance (10%).

## Running it

Open `vit_cifar.ipynb` in Google Colab with a GPU runtime and run all cells. CIFAR-10 and CIFAR-100 download automatically through torchvision. The full 200-epoch runs need a fast GPU.

## Limitations

- "Best test accuracy" picks the best epoch using the test set, so the final-epoch number is the fairer one to quote.
- There is no separate validation split.
