import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import numpy as np
import glob
from PIL import Image
from skimage import metrics
from math import log10, sqrt

device =  'cuda' if torch.cuda.is_available() else 'cpu'

# Parses all the data from the psf files into a numpy array
def getData(file_dir):
    dat_files = glob.glob(file_dir)
    X = []
    Y = []
    for i, filename in enumerate(dat_files):
        if (i+1) % 100 == 0:
            print('Iteration:', i+1)
        with open(filename, 'rb') as file:
            data = file.read()
        data = data.decode('utf-8')
        data = data.split() 
        
        psf_data = []
        while data:
            index = data.index('Size:')
            size = int(data[index + 1])
            psf_data += list(map(float, data[index+2 : index+2+size**2]))
            data = data[index+2+size**2:]
        psf_data = np.array(psf_data).reshape(128, 128, 3)
        Y.append(psf_data)
        
        filename = ''.join(filename.split('\\')[1:])
        parameter = filename.split('_')
        parameter[6] = parameter[6].replace('.dat', '')
        parameter[6] = parameter[6][:-6]
        parameter = parameter[1:]
        X.append(list(map(float, parameter)))
    
    X = np.array(X)
    Y = np.array(Y)

    return X, Y


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(6, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 3072)
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 3, kernel_size=3, padding=1)
        self.conv_transpose1 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2, padding=0)
        self.conv_transpose2 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = x.view(32, 32, 3)
        x = x.permute(2, 0, 1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv_transpose1(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv_transpose2(x))
        x = self.conv5(x)
        x = x.permute(1, 2, 0)
        return x


def Training(X_train, Y_train, X_test, Y_test, num_epochs = 5):
    # Move model to GPU
    net = Net().to(device=device)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(net.parameters())

    for epoch in range(num_epochs):
        running_loss = 0.0
        print('Epoch %d' %(epoch+1))
        for i in range(len(X_train)):
            inputs = torch.from_numpy(X_train[i]).float()
            labels = torch.from_numpy(Y_train[i]).float()
            
            #Move inputs, labels to GPU
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()

            outputs = net(inputs)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print('loss: %.3f' % (running_loss / len(X_train)))
        
        # Evaluate on test set
        with torch.no_grad():
            test_loss = 0.0
            for i in range(len(X_test)):
                inputs = torch.from_numpy(X_test[i]).float()
                labels = torch.from_numpy(Y_test[i]).float()

                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = net(inputs)
                test_loss += criterion(outputs, labels).item()

            print('Test loss: %.3f \n' % (test_loss / len(X_test)))

    torch.save(net.state_dict(), 'model_param2psf.pth')

    return


def Testing(file_dir):
    # Shape of X: (100, 6), Shape of Y: (100, 128, 128, 3) 
    X, Y = getData(file_dir)  
    model = Net()
    model.load_state_dict(torch.load('model_param2psf.pth'))
    
    model.eval()
    # Change data type from numpy to tensor
    X = torch.from_numpy(X).float()
    Y = torch.from_numpy(Y).float()
    
    # Shape of Y, Y_pred: (100, 128 * 128 * 3)
    Y = Y.reshape(len(Y), -1)

    MAE = []
    SSIM = []
    PSNR = []
    for i in range(len(Y)):
        print("Test ", i+1)
        
        Y_pred = model(X[i])
        Y_pred = Y_pred.flatten()
        Y[i] = Y[i].flatten()
        
        # Calculate mae
        diff = abs(Y_pred - Y[i])
        mae = sum(diff) / (128*128*3)
        print("mae:", mae.detach().numpy())
        MAE.append(mae.detach().numpy())
        
        Y_pred_np = Y_pred.detach().numpy()
        Y_i_np = Y[i].detach().numpy()
        
        # Calculate SSIM
        ssim = metrics.structural_similarity(Y_pred_np, Y_i_np, multichannel = True, data_range = 1)
        print("SSIM:", ssim)
        SSIM.append(ssim)
        
        # Calculate PSNR
        mse = np.mean((Y_pred_np - Y_i_np)**2)
        max_pixel = 1.0
        if mse == 0:
            psnr = 100
        else:
            psnr = 20 * log10(max_pixel / sqrt(mse))
        print("PSNR:", psnr)
        PSNR.append(psnr)
        print("\n")
        
    print("Average MAE:", sum(MAE) / len(MAE))
    print("Maximum MAE:", max(MAE))
    print("Minimum MAE:", min(MAE))
    print("\n")
    
    print("Average SSIM:", sum(SSIM) / len(SSIM))
    print("Maximum SSIM:", max(SSIM))
    print("Minimum SSIM:", min(SSIM))
    print("\n")
    
    print("Average PSNR:", sum(PSNR) / len(PSNR))
    print("Maximum PSNR:", max(PSNR))
    print("Minimum PSNR:", min(PSNR))
    print("\n")
    

    img = Image.new('L', (128, 128))

    # Image 1
    img.putdata(Y_pred_np[:128*128])
    img.show()
    # img.save('image1.jpg', format='JPEG')
    img.putdata(Y_i_np[:128*128])
    img.show()
    
    # Image 2
    img.putdata(Y_pred_np[128*128:128*128*2])
    img.show()
    # img.save('image2.jpg', format='JPEG')
    img.putdata(Y_i_np[128*128:128*128*2])
    img.show()

    # Image 3
    img.putdata(Y_pred_np[128*128*2:128*128*3])
    img.show()
    # img.save('image3.jpg', format='JPEG')
    img.putdata(Y_i_np[128*128*2:128*128*3])
    img.show()
    

def main():
    X, Y = getData('train_data/*.dat')
    np.save('data_X', X)
    np.save('data_Y', Y)
    print('Data process complete! \n')
    X_loaded = np.load('data_X.npy')
    Y_loaded = np.load('data_Y.npy')
    X_train, X_test, Y_train, Y_test = train_test_split(X_loaded, Y_loaded, test_size = 0.2)
    Training(X_train, Y_train, X_test, Y_test)
    Testing('test_data/*.dat')

if __name__ == '__main__':
    main()
