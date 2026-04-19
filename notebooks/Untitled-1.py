# %%
import os
import sys

sys.path.append(os.path.abspath(".."))
import torch
import torchvision
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, random_split, Subset
import torch.utils.data as data
import wandb
from data_utils import ApplyTransform, StandardTransform
from torchvision import transforms


# %%
#!pip install lightly

# %%
from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead
from lightly.transforms.simclr_transform import SimCLRTransform

# %%
# Uncomment for running on kaggle
"""
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
wandb_key = user_secrets.get_secret("WANDB_KEY")
wandb.login(key=wandb_key)
"""

# %%
if os.path.exists("/kaggle/working"):
    base_path = "/kaggle/working"

elif "google.colab" in str(get_ipython()):
    base_path = "/content/drive/MyDrive/ProjectSimCLR"

else:
    base_path = "./"

models_path = os.path.join(base_path, "models")

os.makedirs(models_path, exist_ok=True)


def save_checkpoint(model, optimizer, epoch, loss, run_id, hp):

    torch.save(
        {
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "model": model.state_dict(),
            "loss": loss,
            "wandb_run_id": run_id,
            "hp": hp,
        },
        os.path.join(models_path, f"{run_id}_epoch_{epoch}.pth"),
    )


def load_checkpoint(model, optimizer, path, map_location=None):
    if map_location is None:
        if torch.cuda.is_available():
            map_location = "cuda"
        elif torch.backends.mps.is_available():
            map_location = "mps"
        else:
            map_location = "cpu"
    checkpoint = torch.load(path, weights_only=False, map_location=map_location)
    saved_model = checkpoint["model"]
    saved_optimizer = checkpoint["optimizer"]
    model.load_state_dict(saved_model)
    optimizer.load_state_dict(saved_optimizer)
    return checkpoint["epoch"], checkpoint["loss"], checkpoint["wandb_run_id"], checkpoint["hp"]

# %%
class SimCLR(nn.Module):
    def __init__(self, encoder, projection_head):
        super().__init__()
        self.encoder = encoder
        self.projection_head = projection_head

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        x = self.projection_head(x)
        return x

# %%
resnet = torchvision.models.resnet18()
backbone = nn.Sequential(*list(resnet.children())[:-1])
projection_head = SimCLRProjectionHead(input_dim=512, hidden_dim=512, output_dim=128)
model = SimCLR(backbone, projection_head)

# %%
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
model.to(device)

# %%
hp = {
    "batch_size": 256,
    "pipeline_test": False,
    "epochs": 50,
    "temperature": 0.2,
    "optim": {"base_lr": 0.06, "weight_decay": 1e-5},
}

# %%
train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
    ]
)


val_transform = transforms.Compose(
    [
        transforms.Resize(32),
        transforms.ToTensor(),
    ]
)
full_train_dataset = datasets.CIFAR10(root="./data", train=True, download=True)

train_subset, val_subset = random_split(full_train_dataset, [45000, 5000])

train_data = ApplyTransform(train_subset, transform=train_transform)
val_data = ApplyTransform(val_subset, transform=val_transform)

num_workers = 4

if hp["pipeline_test"]:
    train_data = Subset(train_data, range(512))
    val_data = Subset(val_data, range(256))
    hp["epochs"] = 5
    num_workers = 0

# %%
dataloader = DataLoader(
    train_data,
    batch_size=hp["batch_size"],
    shuffle=True,
    drop_last=True,
    num_workers=num_workers,
)

# %%
criterion = NTXentLoss(temperature=hp["temperature"])
optimizer = torch.optim.SGD(
    params=model.named_parameters(),
    lr=hp["optim"]["base_lr"] * hp["batch_size"] / 256,
    weight_decay=hp["optim"]["weight_decay"],
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max = hp["epochs"]
)
# Uncomment when using multiple GPUs
model = torch.nn.DataParallel(model)

# %%
print("Starting Training ...")
run = wandb.init(project="simclr-cifar10", config=hp, mode="online")
model = model.to(device)
for epoch in range(hp["epochs"]):
    print(f"Epoch {epoch} of {hp['epochs']}")
    total_loss = 0
    for batch in dataloader:
        x0, x1 = batch[0]
        x0 = x0.to(device)
        x1 = x1.to(device)
        z0 = model(x0)
        z1 = model(x1)

        loss = criterion(z0, z1)
        total_loss += loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    avg_loss = total_loss / len(dataloader)
    scheduler.step()
    wandb.log({"loss": avg_loss, "epoch": epoch, "lr": scheduler.get_last_lr()[0]})
    if epoch % 10 == 0 and epoch >= 20:
        save_checkpoint(model, optimizer, epoch, loss, run.id, hp)
    print(f"Epoch: {epoch}, Avg_loss: {avg_loss:.5f}")
print("Training Completed, saving the model")
save_checkpoint(model, optimizer, epoch, loss, run.id, hp)

# %%
def linear_evaluation(backbone, train_loader, test_loader, device, epochs = 30):
    # Encoder einfrieren
    for param in backbone.parameters():
        param.requires_grad = False

    classifier = nn.Linear(512, 10).to(device)  # 10 CIFAR-10 Klassen
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    backbone.eval()
    for epoch in range(epochs):
        total_loss = 0
        for imgs, labels in train_loader:
            features = backbone(imgs.to(device))
            features = torch.flatten(features, start_dim=1)
            preds = classifier(features)
            loss = nn.CrossEntropyLoss()(preds, labels.to(device))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss
        print(f"Avg Loss: {total_loss / len(train_loader)}")
        wandb.log({"linear_eval_loss": total_loss / len(train_loader)})

    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            # imgs = imgs[0]
            features = backbone(imgs.to(device))
            features = torch.flatten(features, start_dim=1)
            preds = classifier(features)
            predicted = preds.argmax(dim=1)
            correct += (predicted == labels.to(device)).sum().item()
            total += labels.size(0)
    accuracy = correct / total if total > 0 else 0
    print(f"Accuracy: {accuracy}")
    wandb.log({"linear_eval_accuracy": accuracy})


# %%
if torch.cuda.is_available():
    torch_device = torch.device("cuda")
elif torch.backends.mps.is_available():
    torch_device = torch.device("mps")
else:
    torch_device = torch.device("cpu")

resnet = torchvision.models.resnet18()
backbone = nn.Sequential(*list(resnet.children())[:-1])
projection_head = SimCLRProjectionHead(input_dim=512, hidden_dim=512, output_dim=128)
model = SimCLR(backbone, projection_head)
model = torch.nn.DataParallel(model)
optimizer = torch.optim.SGD(
    params=model.named_parameters(),
    lr=hp["optim"]["base_lr"] * hp["batch_size"] / 256,
    weight_decay=hp["optim"]["weight_decay"],
)
epoch, loss, run_id, hp = load_checkpoint(
    model,
    optimizer,
    "/Users/hannesneumann/Repositories/SelfSupervisedLearning/notebooks/models/g1huy8od_epoch_40.pth",
    map_location=torch_device,
)
model.to(torch_device)
print(model)

# %%
linear_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
        ),
    ]
)
device = torch_device
linear_loader = DataLoader(StandardTransform(train_subset, linear_transform), 128)
val_loader = DataLoader(
    StandardTransform(
        val_subset,
        linear_transform,
    )
)
run = wandb.init(project="simclr-cifar10",  mode="online", id=run_id)
linear_evaluation(model.module.encoder, linear_loader, val_loader, device, epochs=3)

# %%



