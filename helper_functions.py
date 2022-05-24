import matplotlib.pyplot as plt
import numpy as np
import os
from torchvision.io import read_image
import torchvision.transforms.functional as F


def plot_images(root,
                classes):
    image_list = []

    for folder in sorted(os.listdir(root)):
        file = os.listdir(f'{root}/{folder}')[0]
        image = read_image(f'{root}/{folder}/{file}')
        image_list.append(image)

    fig, axs = plt.subplots(ncols=len(image_list),
                            squeeze=False,
                            figsize=(30, 30))

    for i, img in enumerate(image_list):
        img = F.to_pil_image(img)

        axs[i // len(classes), i % len(classes)].imshow(np.asarray(img))
        axs[i // len(classes), i % len(classes)].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        axs[i // len(classes), i % len(classes)].text(img.size[0]/2, 0, f'Label: {classes[i]}', size=20, ha="center",
                                                      fontweight="bold",
                                                      backgroundcolor="white",
                                                      color="black")


def plot_batch(img):
    npimg = img.numpy()
    plt.figure(figsize=[15, 15])
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')


def plot_images_predictions(
        model,
        classes,
        dataset,
        device):
    image_list = []
    labels = []
    preds = []

    for i, (x, y) in enumerate(dataset):
        image = read_image(dataset.imgs[i][0])
        image_list.append(image)
        labels.append(y)

        x = x.to(device)
        pred = model(x.unsqueeze(0))
        preds.append(pred.argmax(1))

    fig, axs = plt.subplots(ncols=len(classes),
                            nrows=(len(image_list) // len(classes)) + 1,
                            squeeze=False,
                            figsize=(50, 50))

    for i, img in enumerate(image_list):
        img = F.to_pil_image(img)

        axs[i // len(classes), i % len(classes)].imshow(np.asarray(img))
        axs[i // len(classes), i % len(classes)].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        axs[i // len(classes), i % len(classes)].text(img.width / 2, img.height * 0.1, f'Label: {classes[labels[i]]}', size=20, ha="center",
                                fontweight="bold",
                                backgroundcolor="white",
                                color="black")

        axs[i // len(classes), i % len(classes)].text(img.width / 2, img.height * 0.9, f'Predicted: {classes[preds[i]]}', size=20, ha="center",
                                fontweight="bold",
                                backgroundcolor="green" if classes[preds[i]] == classes[labels[i]] else "red",
                                color="white")

    for j in range(i % len(classes), len(classes)):
        axs[-1, j].axis('off')


def plot_train_val_loss(train_losses,
                        val_losses):
    figure = plt.figure(figsize=(15, 5))
    plt.plot(train_losses, label='Train loss')
    plt.plot(val_losses, label='Test loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")