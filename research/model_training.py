import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_CIFAR(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_CIFAR, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet44(num_classes=10):
    # For ResNet-n: n = 6n_layers + 2 → n_layers=7 → ResNet-44
    return ResNet_CIFAR(BasicBlock, [7, 7, 7], num_classes)


def fgsm_attack_on_normalized(x_norm, y, model, epsilon, mean, std, device):
    """
    x_norm: normalized images (batch)
    y: labels
    model: eval model
    epsilon: float in pixel-space (0..1)
    mean,std: tuples for Normalize
    returns: perturbed normalized images (clamped so pixel values remain in [0,1])
    """
    # convert epsilon (pixel-space) to normalized-space per channel
    std_tensor = torch.tensor(std, device=device).view(1, -1, 1, 1)
    eps_norm = epsilon / std_tensor  # broadcastable

    # need gradients w.r.t. input; clone to avoid in-place issues
    x_norm_adv = x_norm.clone().detach().requires_grad_(True)

    # forward + loss
    out = model(x_norm_adv)
    loss = nn.CrossEntropyLoss()(out, y)
    model.zero_grad()
    loss.backward()

    # sign of gradient in normalized-space
    grad_sign = x_norm_adv.grad.data.sign()

    # take a single step (FGSM) in normalized space then clamp pixel values
    x_norm_pert = x_norm + eps_norm * grad_sign

    # To clamp properly we convert allowed pixel range [0,1] into normalized range then clamp there
    mean_tensor = torch.tensor(mean, device=device).view(1, -1, 1, 1)
    std_tensor = std_tensor  # already defined
    min_norm = (0.0 - mean_tensor) / std_tensor
    max_norm = (1.0 - mean_tensor) / std_tensor
    x_norm_pert = torch.max(torch.min(x_norm_pert, max_norm), min_norm)

    return x_norm_pert.detach()


def epoch_adversarial(model, loader, epsilon, mean, std, device):
    model.eval()
    total_correct = 0
    total = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        # generate perturbation and evaluate
        X_adv = fgsm_attack_on_normalized(X, y, model, epsilon, mean, std, device)
        with torch.no_grad():
            out = model(X_adv)
            pred = out.max(1)[1]
            total_correct += pred.eq(y).sum().item()
            total += y.size(0)
    acc = 100.0 * total_correct / total
    return acc


def imshow(img):
    img = img.cpu().numpy().transpose((1, 2, 0))
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    plt.axis('off')

def visualize_fgsm(model, loader, epsilon, mean, std, device):
    model.eval()
    data_iter = iter(loader)
    images, labels = next(data_iter)
    images, labels = images.to(device), labels.to(device)

    adv_images = fgsm_attack_on_normalized(images, labels, model, epsilon, mean, std, device)
    perturbations = adv_images - images

    # Unnormalize for display
    mean_tensor = torch.tensor(mean, device=device).view(1, -1, 1, 1)
    std_tensor = torch.tensor(std, device=device).view(1, -1, 1, 1)

    orig_unnorm = images * std_tensor + mean_tensor
    adv_unnorm = adv_images * std_tensor + mean_tensor
    pert_vis = perturbations / (2*epsilon) + 0.5  # normalize to [0,1] for plotting

    outputs_orig = model(images)
    preds_orig = outputs_orig.max(1)[1]
    outputs_adv = model(adv_images)
    preds_adv = outputs_adv.max(1)[1]

    plt.figure(figsize=(12, 6))
    for i in range(6):  # show first 6 samples
        plt.subplot(3, 6, i+1)
        imshow(orig_unnorm[i])
        plt.title(f"Orig: {preds_orig[i].item()}\nTrue: {labels[i].item()}")
        plt.subplot(3, 6, i+1+6)
        imshow(pert_vis[i])
        plt.title("Perturb")
        plt.subplot(3, 6, i+1+12)
        imshow(adv_unnorm[i])
        plt.title(f"Adv: {preds_adv[i].item()}")
    plt.tight_layout()
    plt.show()

def main():
    # --- (same data, model, optimizer, scheduler setup as before) ---
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                             shuffle=False, num_workers=2)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model = ResNet44(num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1,
                          momentum=0.9, weight_decay=5e-4)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[100, 150], gamma=0.1
    )

    # training and test functions (same as before) ...
    def train(epoch):
        model.train()
        running_loss = 0.0
        total, correct = 0, 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        acc = 100.*correct/total
        print(f"Epoch {epoch} Train Loss: {running_loss/len(trainloader):.3f} | Train Acc: {acc:.2f}%")

    def test(epoch):
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        acc = 100.*correct/total
        print(f"Epoch {epoch} Test Loss: {test_loss/len(testloader):.3f} | Test Acc: {acc:.2f}%")
        return acc
    
    def train_mixed(epoch, eps_list, mean, std):
        model.train()
        running_loss = 0.0
        total, correct = 0, 0

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            # start with clean samples
            all_inputs = [inputs]
            all_targets = [targets]

            # generate adversarial samples for each epsilon
            for eps in eps_list:
                adv_inputs = fgsm_attack_on_normalized(inputs, targets, model, eps, mean, std, device)
                all_inputs.append(adv_inputs)
                all_targets.append(targets)

            # concatenate clean + all adversarial versions
            mixed_inputs = torch.cat(all_inputs, dim=0)
            mixed_targets = torch.cat(all_targets, dim=0)

            # forward + backward
            outputs = model(mixed_inputs)
            loss = criterion(outputs, mixed_targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += mixed_targets.size(0)
            correct += predicted.eq(mixed_targets).sum().item()

        acc = 100.*correct/total
        print(f"[Mixed Adv Train] Epoch {epoch} Loss: {running_loss/len(trainloader):.3f} | Acc: {acc:.2f}%")

    def test_all(epoch, eps_list, mean, std, device):
        model.eval()
        # clean accuracy
        clean_acc = test(epoch)

        # adversarial accuracy at each epsilon
        adv_accs = {}
        for eps in eps_list:
            acc = epoch_adversarial(model, testloader, eps, mean, std, device)
            adv_accs[eps] = acc

        return min(clean_acc, *adv_accs.values())



    # ---- Train ----

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    eps_list = [0.0, 0.01, 0.03, 0.05]

    best_acc = 0
    for epoch in range(0):
       train_mixed(epoch, eps_list, mean, std)
       acc = test_all(epoch, eps_list, mean, std, device)
       scheduler.step()

       if acc > best_acc:
           torch.save(model.state_dict(), 'best_resnet44_cifar10_robust.pth')
           best_acc = acc

    print(f"Best Test Accuracy: {best_acc:.2f}%")

    # ---- Load best model if available and run FGSM eval ----
    try:
        model.load_state_dict(torch.load('best_resnet44_cifar10.pth', map_location=device))
        print("Loaded best_resnet44_cifar10.pth for adversarial evaluation.")
    except Exception as e:
        print("Could not load checkpoint, using current model. Error:", e)

    # pixel-space epsilons to try (0..1)
    eps_list = [0.0, 0.005, 0.01, 0.03, 0.1]  # tweak as you want
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    for eps in eps_list:
        visualize_fgsm(model, testloader, eps, mean, std, device)
        acc_adv = epoch_adversarial(model, testloader, eps, mean, std, device)
        print(f"FGSM eps={eps:.4f} -> Test Accuracy: {acc_adv:.2f}%")

if __name__ == "__main__":
    main()
