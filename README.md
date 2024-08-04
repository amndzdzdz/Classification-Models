# Image Classification Models

Welcome to my repository where I've implemented some well-known image classification models: **AlexNet**, **VGGNet**, and **ResNet**. This is one of my first projects, so I'm excited to share it with you!

## What's Included?

- **AlexNet**: A classic convolutional neural network that made a big impact in 2012.
- **VGGNet**: A simple yet powerful model, with versions that have 16 or 19 layers.
- **ResNet**: Known for its skip connections, this model can be easily modified to create deeper versions like ResNet-50 or ResNet-101.

## Features

- **PCA Color Augmentation**: I’ve included PCA color augmentation during preprocessing, similar to what’s used in the original AlexNet Paper. This helps in making the training data more diverse and robust.

## Limitations

I wasn't able to fully train the models due to hardware limitations, but I did test the training process to ensure it runs and that the loss is reasonable. So while the models aren’t fully trained, the setup is here for you to use or build upon.

## How to Use

To get started with this code:

1. Clone the repository:
    ```bash
    git clone https://github.com/amndzdzdz/Classification-Models.git
    ```

2. Install the necessary dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Modify the code as needed, and try it out on your own datasets.

## ResNet Customization

If you want a deeper version of ResNet, you can easily modify the architecture by changing the layer configuration in the code. It’s set up to allow easy expansion to ResNet-101, ResNet-152, or beyond.

## Credits

If you find this code helpful, please consider giving me credit by linking back to this repository or mentioning my name. I’d really appreciate it!

---

**Author:** Amin Dziri

