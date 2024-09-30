import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import pickle

plt.style.use('seaborn-v0_8-whitegrid')
COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']
import matplotlib
matplotlib.rcParams.update({'font.size':24})

# ------------------------------------------------------------------------------

N_EPOCHS = 250
N_CL_EPOCHS = 10
N_SEEDS = 5
SAVE_EVERY = 10
BATCH_SIZE_TRAIN = 64
BATCH_SIZE_TEST = 1000
TRAIN_LOADER_ITERS = 100
LEARNING_RATE = 0.001
FC_DIM = 32
ETA_IN = np.array([0,4,8,12,16,20,24,28,32])
MODELS = np.concatenate((
			            ['SNN','DNN']
                        [f'SHANN_{inx}' for inx in ETA_IN]
                        ))
LOG_INTERVAL = 5
DATASET = 'MNIST' #'MNIST', 'FashionMNIST', 'CIFAR10'
DATASET_FOLDER = 'datasets/'
RESULT_FOLDER = 'results/'
MODEL_FOLDER = 'models/'

#RANDOM_SEED = 42
#torch.backends.cudnn.enabled = False
#torch.manual_seed(RANDOM_SEED)

# ------------------------------------------------------------------------------

class DNN(nn.Module):
    def __init__(self, fc_dim=128, eta_out=0):
        super(DNN,self).__init__()
        self.fc1 = nn.Linear(28*28, fc_dim) 
        self.fc2 = nn.Linear(fc_dim, fc_dim)
        self.fc3 = nn.Linear(fc_dim, fc_dim)
        self.fc4 = nn.Linear(fc_dim, fc_dim+3*eta_out)
        self.relu = nn.ReLU()

    def forward(self, img): #convert + flatten
        x = img.view(-1, 28*28)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        return x


class SNN(nn.Module):
    def __init__(self, fc_dim=128, eta_out=0):
        super(SNN,self).__init__()
        self.fc1 = nn.Linear(28*28, fc_dim+3*eta_out) 
        self.relu = nn.ReLU()
        
    def forward(self, img): #convert + flatten
        x = img.view(-1, 28*28)
        x = self.relu(self.fc1(x))
        return x


class SHANN(nn.Module):
    def __init__(self, fc_dim=128, eta_in=16, eta_out=0):
        super(SHANN,self).__init__()
        self.fc_dim = fc_dim
        self.eta_in = eta_in
        self.eta_out = eta_out
        self.relu = nn.ReLU()

        # From input to hidden layers
        self.fc_in_1 = nn.Linear(28*28, self.fc_dim)
        if self.eta_in > 0:
            self.fc_in_2 = nn.Linear(28*28, self.eta_in)
            self.fc_in_3 = nn.Linear(28*28, self.eta_in)
            self.fc_in_4 = nn.Linear(28*28, self.eta_in)

        # Between hidden layers
        self.fc_1_2 = nn.Linear(self.fc_dim-self.eta_out, self.fc_dim-self.eta_in)
        self.fc_2_3 = nn.Linear(self.fc_dim-self.eta_out, self.fc_dim-self.eta_in)
        self.fc_3_4 = nn.Linear(self.fc_dim-self.eta_out, self.fc_dim-self.eta_in)


    def forward(self, img): #convert + flatten
        x_in = img.view(-1, 28*28)

        # Hidden layer 1
        x_1 = self.relu(self.fc_in_1(x_in))
        if self.eta_out > 0:
            x_1_hidden, x_1_out = torch.split(x_1, [self.fc_dim-self.eta_out, self.eta_out], dim=-1)
        else:
            x_1_hidden = x_1

        # Hidden layer 2
        if self.eta_in > 0:
            x_in_2 = self.relu(self.fc_in_2(x_in))
            x_1_2 = self.relu(self.fc_1_2(x_1_hidden))
            x_2 = torch.cat([x_in_2,x_1_2], dim=-1)
        else:
            x_2 = self.relu(self.fc_1_2(x_1_hidden))

        if self.eta_out > 0:
            x_2_hidden, x_2_out = torch.split(x_2, [self.fc_dim-self.eta_out, self.eta_out], dim=-1)
        else:
            x_2_hidden = x_2

        # Hidden layer 3
        if self.eta_in > 0:
            x_in_3 = self.relu(self.fc_in_3(x_in))
            x_2_3 = self.relu(self.fc_2_3(x_2_hidden))
            x_3 = torch.cat([x_in_3,x_2_3], dim=-1)
        else:
            x_3 = self.relu(self.fc_2_3(x_2_hidden))

        if self.eta_out > 0:
            x_3_hidden, x_3_out = torch.split(x_3, [self.fc_dim-self.eta_out, self.eta_out], dim=-1)
        else:
            x_3_hidden = x_3

        # Hidden layer 4
        if self.eta_in > 0:
            x_in_4 = self.relu(self.fc_in_4(x_in))
            x_3_4 = self.relu(self.fc_3_4(x_3_hidden))
            x_4 = torch.cat([x_in_4,x_3_4], dim=-1)
        else:
            x_4 = self.relu(self.fc_3_4(x_3_hidden))

        if self.eta_out > 0:
            x = torch.cat([x_1_out,x_2_out,x_3_out,x_4], dim=-1)
        else:
            x = x_4
        return self.relu(x_4)


class Decoder(nn.Module):
    def __init__(self, fc_dim=128, eta_out=16):
        super(Decoder,self).__init__()
        self.fc = nn.Linear(fc_dim+3*eta_out, 28*28)
        
    def forward(self, x): #convert + unflatten
        x = torch.sigmoid(self.fc(x))
        x = torch.unflatten(x, dim=-1, sizes=(1,28,28))
        return x


class Classifier(nn.Module):
    def __init__(self, fc_dim=128, eta_out=16):
        super(Classifier,self).__init__()
        self.fc = nn.Linear(fc_dim+3*eta_out, 10)
       
    def forward(self, x): #convert + unflatten
        return F.log_softmax(self.fc(x), dim=-1)

# ------------------------------------------------------------------------------

def plot_all_accuracies(model_accuracies, display=True):

    fig,ax = plt.subplots(figsize=(12,12))

    for in_idx, eta_in in enumerate(ETA_IN):
            model = 'SHANN_'+str(eta_in)
            accuracies = model_accuracies[model]
            shann_mean = np.mean(accuracies, axis=0)
            shann_std = np.std(accuracies, axis=0)
            x = np.arange(len(shann_mean))

            ax.fill_between(x,
                                    shann_mean - shann_std,
                                    shann_mean + shann_std,
                                    color=COLORS[in_idx], 
                                    alpha=0.1, zorder=2)
            ax.plot(x, shann_mean,
                            color=COLORS[in_idx],
                            linewidth=2, label=model, zorder=4)

            ax.legend()
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Accuracy')
    if display:
        plt.show()

    fig.savefig(f'{RESULT_FOLDER}accuracies.png',dpi=300)

def plot_reconstruction_errors(model_errors, display=True):

    fig,ax = plt.subplots(figsize=(12,12))

    for in_idx, eta_in in enumerate(ETA_IN):
            model = 'SHANN_'+str(eta_in)
            errors = model_errors[model]
            shann_mean = np.mean(errors, axis=0)
            shann_std = np.std(errors, axis=0)
            x = np.arange(len(shann_mean))

            ax.fill_between(x,
                                    shann_mean - shann_std,
                                    shann_mean + shann_std,
                                    color=COLORS[in_idx], 
                                    alpha=0.1, zorder=2)
            ax.plot(x, shann_mean,
                            color=COLORS[in_idx],
                            linewidth=2, label=model, zorder=4)

            ax.legend()
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Reconstruction error')
    if display:
        plt.show()

    fig.savefig(f'{RESULT_FOLDER}reconstruction_errors.png',dpi=300)

def train_ae(network, decoder, optimizer, epoch, loader, display=False):
    network.train()
    decoder.train()
    mse_loss = nn.MSELoss()
    epoch_losses = []
    loader_iterator = iter(loader)
    for batch_idx in range(TRAIN_LOADER_ITERS):
        (data, _) = next(loader_iterator)
        optimizer.zero_grad()
        latent = network(data)
        reconstruction = decoder(latent)
        loss = mse_loss(data, reconstruction)
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            if display:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(loader.dataset),
                    100. * batch_idx / len(loader), loss.item()))
            epoch_losses.append(loss.item())
    return epoch_losses

def test_ae(network, decoder, loader, display=False):
    network.eval()
    decoder.eval()
    mse_loss = nn.MSELoss()
    test_loss = 0
    correct = 0
    epoch_losses = []
    with torch.no_grad():
        for data, _ in loader:
            latent = network(data)
            reconstruction = decoder(latent)
            test_loss += mse_loss(data, reconstruction).item()
    test_loss /= len(loader.dataset)
    epoch_losses.append(test_loss)
    if display:
        print('\nTest set: Avg. loss: {:.4f}\n'.format(test_loss))
    return epoch_losses


def train_cl(network, classifier, optimizer, epoch, loader, display=False):
    network.train()
    cross_entropy_loss = nn.CrossEntropyLoss()
    epoch_losses = []
    loader_iterator = iter(loader)
    for batch_idx in range(TRAIN_LOADER_ITERS):
        (data, target) = next(loader_iterator)
        optimizer.zero_grad()
        latent = network(data)
        prediction = classifier(latent)
        loss = cross_entropy_loss(prediction, target)
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            if display:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(loader.dataset),
                    100. * batch_idx / len(loader), loss.item()))
            epoch_losses.append(loss.item())
    return epoch_losses

def test_cl(network, classifier, loader, display=False):
    network.eval()
    cross_entropy_loss = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    epoch_losses = []
    with torch.no_grad():
        for data, target in loader:
            latent = network(data)
            prediction = classifier(latent)
            test_loss += cross_entropy_loss(prediction, target).item()
            pred = prediction.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    accuracy = 100. * correct / len(loader.dataset)
    test_loss /= len(loader.dataset)
    epoch_losses.append(test_loss)
    if display:
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(loader.dataset), accuracy.detach().numpy()))
    return epoch_losses, [accuracy.detach().numpy()]

# ------------------------------------------------------------------------------

def main():

    if DATASET == 'MNIST':
        train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(
                                                        DATASET_FOLDER, train=True, download=True,
                                                        transform=torchvision.transforms.Compose([
                                                            torchvision.transforms.ToTensor(),
                                                        ])),
                                                batch_size=BATCH_SIZE_TRAIN, shuffle=True)

        test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(
                                                        DATASET_FOLDER, train=False, download=True,
                                                        transform=torchvision.transforms.Compose([
                                                            torchvision.transforms.ToTensor(),
                                                        ])),
                                                batch_size=BATCH_SIZE_TEST, shuffle=True)

    elif DATASET == 'FashionMNIST':
        train_loader = torch.utils.data.DataLoader(torchvision.datasets.FashionMNIST(
                                                        DATASET_FOLDER, train=True, download=True,
                                                        transform=torchvision.transforms.Compose([
                                                            torchvision.transforms.ToTensor(),
                                                        ])),
                                                batch_size=BATCH_SIZE_TRAIN, shuffle=True)

        test_loader = torch.utils.data.DataLoader(torchvision.datasets.FashionMNIST(
                                                        DATASET_FOLDER, train=False, download=True,
                                                        transform=torchvision.transforms.Compose([
                                                            torchvision.transforms.ToTensor(),
                                                        ])),
                                                batch_size=BATCH_SIZE_TEST, shuffle=True)

    elif DATASET == 'CIFAR10':
        train_loader = torch.utils.data.DataLoader(torchvision.datasets.CIFAR10(
                                                        DATASET_FOLDER, train=True, download=True,
                                                        transform=torchvision.transforms.Compose([
                                                            torchvision.transforms.ToTensor(),
                                                            torchvision.transforms.CenterCrop(28),
                                                            torchvision.transforms.Grayscale(1),
                                                        ])),
                                                batch_size=BATCH_SIZE_TRAIN, shuffle=True)

        test_loader = torch.utils.data.DataLoader(torchvision.datasets.CIFAR10(
                                                        DATASET_FOLDER, train=False, download=True,
                                                        transform=torchvision.transforms.Compose([
                                                            torchvision.transforms.ToTensor(),
                                                            torchvision.transforms.CenterCrop(28),
                                                            torchvision.transforms.Grayscale(1),
                                                        ])),
                                                batch_size=BATCH_SIZE_TEST, shuffle=True)

    model_errors = {model: [[] for _ in range(N_SEEDS)] for model in MODELS}
    model_accuracies = {model: [[] for _ in range(N_SEEDS)] for model in MODELS}

    for model in MODELS:
        print()
        print('Model',model)
        for seed_idx in range(N_SEEDS):

            train_losses = []
            test_losses = []
            accuracies = []

            if model[:3] == 'SNN':
                network = SNN(fc_dim=FC_DIM)
            elif model[:3] == 'DNN':
                network = DNN(fc_dim=FC_DIM)
            elif model[:5] == 'SHANN':
                _, eta_in = model.split('_')
                network = SHANN(fc_dim=FC_DIM, eta_in=int(eta_in))

            decoder = Decoder(fc_dim=FC_DIM)
            classifier = Classifier(fc_dim=FC_DIM)
	
            optimizer = optim.Adam(
                            list(network.parameters()) + list(decoder.parameters()),
                            lr=LEARNING_RATE
                            )
            cl_optimizer = optim.Adam(classifier.parameters(), lr=LEARNING_RATE)                
            
            test_epoch_loss = test_ae(network, decoder, loader=test_loader, display=False)
            test_losses = np.concatenate([test_losses,test_epoch_loss])
            _,accuracy = test_cl(network, classifier, loader=test_loader, display=False)
            accuracies = np.concatenate([accuracies,accuracy])
            
            for epoch in tqdm(range(0, N_EPOCHS + 1), leave=False):
                # Train
                train_epoch_losses = train_ae(network, decoder, optimizer, epoch=epoch, loader=train_loader, display=False)
                train_losses = np.concatenate([train_losses,train_epoch_losses])
                
                if epoch % SAVE_EVERY == 0:
                
                	#Test 
                	test_epoch_loss = test_ae(network, decoder, loader=test_loader, display=False)
                	test_losses = np.concatenate([test_losses,test_epoch_loss])
                    
                    classifier = Classifier(fc_dim=FC_DIM)
                    cl_optimizer = optim.Adam(classifier.parameters(), lr=LEARNING_RATE)
                    
                    for cl_epoch in range(N_CL_EPOCHS):
                    	_ = train_cl(network, classifier, cl_optimizer, epoch=cl_epoch, loader=train_loader, display=False)
                    _, accuracy = test_cl(network, classifier, loader=test_loader, display=False)
                    accuracies = np.concatenate([accuracies,accuracy])
                    
                    filename = f"{MODEL_FOLDER}{model}_{FC_DIM}dim_{seed_idx}seed_{epoch}epochs"
                    torch.save(network.state_dict(), f'{filename}_net.pth')
                    torch.save(decoder.state_dict(), f'{filename}_dec.pth')
                    torch.save(classifier.state_dict(), f'{filename}_cl.pth')
                                  
            model_errors[model][seed_idx] = test_losses
            model_accuracies[model][seed_idx] = accuracies

    filename = RESULT_FOLDER+f'{DATASET}_{N_EPOCHS}epochs_{N_SEEDS}seeds_{FC_DIM}dim'
    with open(filename+'ERRORS.pkl', 'wb') as f:
        pickle.dump(model_errors, f)
    with open(filename+'ACCURACIES.pkl', 'wb') as f:
        pickle.dump(model_accuracies, f)

    plot_reconstruction_errors(model_errors)
    plot_all_accuracies(model_accuracies)
    

if __name__ == '__main__':
    main()
