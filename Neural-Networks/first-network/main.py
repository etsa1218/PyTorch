import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import numpy as np

transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5), (0.5))])
train_set = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=32, shuffle=False)


print(f'# of images in training set is {len(train_set)}')
print(f'# of images in the test set is {len(test_set)}')

print(
    f'Shape of the images in the training set: {train_loader.dataset[0][0].shape}')

fig, axes = plt.subplots(1, 10, figsize=(12, 3))
for i in range(10):
    axes[i].imshow(train_loader.dataset[i][0].squeeze(), cmap='gray')
    axes[i].set_title(train_loader.dataset[i][1])
    axes[i].axis('off')
plt.show()


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.input = nn.Linear(28*28, 128)
        self.hidden = nn.Linear(128, 64)
        self.output = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.input(x))
        x = F.relu(self.hidden(x))
        x = F.log_softmax(self.output(x), dim=1)
        return x


model = NeuralNetwork()

loss_function = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 5
for epoch in range(epochs):
    for images, labels in train_loader:
        optimizer.zero_grad()

        output = model(images)
        loss = loss_function(output, labels)

        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')


def view_classify(image, probabilities):
    probabilities = probabilities.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6, 9), ncols=2)
    ax1.imshow(image.numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), probabilities)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()


images, _ = next(iter(test_loader))

image = images[0]
with torch.no_grad():
    log_probabilities = model(image)

probabilities = torch.exp(log_probabilities)
view_classify(image.view(1, 28, 28), probabilities)

plt.show()
