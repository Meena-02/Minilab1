""" 
Test A:
Model Configuration Changes
    ~ All layers frozen
    ~ Fully Connected layer changed to match CIFAR-10
Loss Function Used:
    ~ Cross Entropy Loss
Optimizer Used:
    ~ Adam with Learning rate 0.001
Regularization Used:
    ~ None
"""
import torch
import torchvision.models as models
import helper_func as hf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 5

model = models.resnet50(pretrained=True)

for param in model.parameters():
    param.requires_grad = False
    
num_classes = 10
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
trainset, trainloader, valset, valloader, testset, testloader = hf.load_dataset()

loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

print(model)
model.to(device)

for epoch in range(num_epochs):
    print(f'Training [{epoch+1}/{num_epochs}]')
    model, train_loss, train_acc, val_loss, val_acc = hf.train(model, trainloader, len(trainset),
                                                            valloader, len(valset), loss_function,
                                                            optimizer, device)
    print(f"Epoch [{epoch+1}/{num_epochs}], train loss: {train_loss:.4f}, train acc: {train_acc:.4f}, val loss: {val_loss:.4f}, val acc: {val_acc:.4f}")

torch.save(model.state_dict(), f'models/resnet50_cifar10_test_A.pth')
print(f"Finished training the model. Model has been saved")

test_loss, test_acc = hf.test(model, testloader, len(testset), loss_function, "test_A", device)
print(f'Test loss: {test_loss:.2f}, Test Accuracy: {test_acc:.2f}')

