import torch 
import pandas as pd 
 
# Загрузка входных данных из файла 
input_data = pd.read_csv('input.csv') 
 
# Преобразование входных данных в тензор PyTorch 
X_input = input_data.values 
X_input_tensor = torch.Tensor(X_input) 
 
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
 
input_size = X_input.shape[1] 
hidden_size1 = 16 
hidden_size2 = 12 
hidden_size3 = 8 
hidden_size4 = 6 
output_size = 1 

# Создание экземпляра модели и загрузка весов из сохраненного файла 
loaded_model = ImprovedNeuralNet(input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, output_size) 
loaded_model.load_state_dict(torch.load('model.pth')) 
loaded_model.eval()  # Убедитесь, что модель в режиме eval() 
 
# Получение предсказания от загруженной модели 
with torch.no_grad(): 
    predicted_output = loaded_model(X_input_tensor) 
    print(predicted_output.numpy())  # Вывод предсказанных значений