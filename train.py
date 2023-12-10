import torch 
import torch.nn as nn 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
 
data = pd.read_csv('input.csv') 
 
X = data.iloc[:, :-1].values 
y = data.iloc[:, -1].values 
 
scaler = StandardScaler() 
X = scaler.fit_transform(X) 
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
 
X_train_tensor = torch.Tensor(X_train) 
y_train_tensor = torch.Tensor(y_train) 
X_test_tensor = torch.Tensor(X_test) 
y_test_tensor = torch.Tensor(y_test) 
 
class ImprovedNeuralNet(torch.nn.Module): 
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, output_size): 
        super(ImprovedNeuralNet, self).__init__() 
        self.fc1 = torch.nn.Linear(input_size, hidden_size1) 
        self.relu1 = torch.nn.ReLU() 
        self.bn1 = torch.nn.BatchNorm1d(hidden_size1)  # Batch Normalization 
        self.fc2 = torch.nn.Linear(hidden_size1, hidden_size2) 
        self.relu2 = torch.nn.ReLU() 
        self.bn2 = torch.nn.BatchNorm1d(hidden_size2)  # Batch Normalization 
        self.dropout = torch.nn.Dropout(0.01)  # Dropout 
        self.fc3 = torch.nn.Linear(hidden_size2, hidden_size3) 
        self.relu3 = torch.nn.ReLU() 
        self.bn3 = torch.nn.BatchNorm1d(hidden_size3)  # Batch Normalization 
        self.fc4 = torch.nn.Linear(hidden_size3, hidden_size4) 
        self.relu4 = torch.nn.ReLU() 
        self.bn4 = torch.nn.BatchNorm1d(hidden_size4)  # Batch Normalization 
        self.fc5 = torch.nn.Linear(hidden_size4, output_size) 
 
    def forward(self, x): 
        out = self.fc1(x) 
        out = self.relu1(out) 
        out = self.bn1(out) 
        out = self.fc2(out) 
        out = self.relu2(out) 
        out = self.bn2(out) 
        out = self.dropout(out) 
        out = self.fc3(out) 
        out = self.relu3(out) 
        out = self.bn3(out) 
        out = self.fc4(out) 
        out = self.relu4(out) 
        out = self.bn4(out) 
        out = self.fc5(out) 
        return out 
 
input_size = X.shape[1] 
hidden_size1 = 16 
hidden_size2 = 12 
hidden_size3 = 8 
hidden_size4 = 6 
output_size = 1 
 
model = ImprovedNeuralNet(input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, output_size) 
 
criterion = nn.MSELoss() 
 
optimizer = torch.optim.Adam(model.parameters(), lr=0.1) 
 
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1) 
 
num_epochs = 2500 
for epoch in range(num_epochs): 
    outputs = model(X_train_tensor) 
    loss = criterion(outputs, y_train_tensor.view(-1, 1)) 
 
    optimizer.zero_grad() 
    loss.backward() 
    optimizer.step() 
 
    if (epoch+1) % 10 == 0: 
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}') 
 
with torch.no_grad(): 
    predicted = model(X_test_tensor) 
    test_loss = criterion(predicted, y_test_tensor.view(-1, 1)) 
 
print(f'Test Loss: {test_loss.item():.4f}') 
 
torch.save(model.state_dict(), 'model2.pth')