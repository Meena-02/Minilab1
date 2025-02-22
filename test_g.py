""" 
Test F:
Model Configuration Changes
    ~ Layer 4 unfrozen
    ~ Fully Connected layer changed to match CIFAR-10
Loss Function Used:
    ~ Cross Entropy Loss
Optimizer Used:
    ~ Adam with Learning rate 0.001
Regularization Used:
    ~ None
Scheduler Used:
    ~ Cosine Annealing with T max 5 and eta_min 0.0001
Batch size:
    ~ Different batch size 
"""
import torch
import torchvision.models as models
import helper_func as hf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 5
batch_size = 64

model = models.resnet50(pretrained=True)

for param in model.parameters():
    param.requires_grad = False
    
for param in model.layer4.parameters():
    param.requires_grad = True
    
num_classes = 10
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
trainset, trainloader, valset, valloader, testset, testloader = hf.load_dataset(batch_size)

loss_function = torch.nn.CrossEntropyLoss()
params_to_update = list(model.layer4.parameters()) + list(model.fc.parameters())
optimizer = torch.optim.Adam(params_to_update, lr=0.001)

step_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

print(model)
model.to(device)

for epoch in range(num_epochs):
    print(f'Training [{epoch+1}/{num_epochs}]')
    model, train_loss, train_acc, val_loss, val_acc = hf.train(model, trainloader, len(trainset),
                                                            valloader, len(valset), loss_function,
                                                            optimizer, device)
    step_lr_scheduler.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], batch_size: {batch_size}, train loss: {train_loss:.4f}, train acc: {train_acc:.4f}, val loss: {val_loss:.4f}, val acc: {val_acc:.4f}")
    print(f'Current learning rate: {step_lr_scheduler.get_last_lr()[0]:.10f}')
    
torch.save(model.state_dict(), f'models/resnet50_cifar10_test_G_64.pth')
print(f"Finished training the model. Model has been saved")

test_loss, test_acc = hf.test(model, testloader, len(testset), loss_function, "test_G_64" ,device)
print(f'Test loss: {test_loss:.2f}, Test Accuracy: {test_acc:.2f}')

