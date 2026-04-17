import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

train_dataset=datasets.MNIST(root='./data',train=True,transform=transforms.ToTensor(),download=True)
train_loader=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=128,shuffle=True)

class MADE(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        super(MADE,self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        m_input=torch.arange(input_dim)
        m_hidden=torch.randint(0,input_dim-1,(hidden_dim,))
        m_out=torch.arange(output_dim)

        mask1=(m_hidden.unsqueeze(1)>=m_input.unsqueeze(0))
        mask2=(m_out.unsqueeze(1)>m_hidden.unsqueeze(0))

        self.register_buffer('mask1',mask1)
        self.register_buffer('mask2',mask2)

        self.fc1 = nn.Linear(self.input_dim,self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim,self.output_dim)

    def forward(self,x):
        h=F.linear(x,self.fc1.weight*self.mask1,self.fc1.bias)
        h=F.relu(h)
        return F.sigmoid(F.linear(h,self.fc2.weight*self.mask2,self.fc2.bias))



model=MADE(784,500,784)
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)

for epoch in range(30):
    total_loss=0
    for images,labels in train_loader:
        x=images.view(-1,784)
        x=(x>0.5).float()
        output=model(x)
        loss=F.binary_cross_entropy(output,x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")


def sample(model, num_samples=16):
    model.eval()
    with torch.no_grad():
        x = torch.zeros(num_samples, 784)
        for i in range(784):
            out = model(x)           # forward pass
            p = out[:, i]            # probability for pixel i
            x[:, i] = torch.bernoulli(p)  # sample 0 or 1
    return x

samples = sample(model)

# Visualize
import matplotlib.pyplot as plt
fig, axes = plt.subplots(4, 4, figsize=(6, 6))
for idx, ax in enumerate(axes.flat):
    ax.imshow(samples[idx].view(28, 28), cmap='gray')
    ax.axis('off')
plt.tight_layout()
plt.savefig('made_samples.png')
plt.show()