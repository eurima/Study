from sklearn.datasets import load_boston
import torch
import torch.nn as nn
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print("torch :", torch.__version__, "DEVICE :",DEVICE)

#1.데이터
datasets = load_boston()

x = datasets.data
y = datasets.target

x = torch.FloatTensor(x)
y = torch.FloatTensor(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, 
         train_size = 0.7, shuffle = True, random_state = 66)

# x_train = torch.FloatTensor(x_train)
# x_test = torch.FloatTensor(x_test)
y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
y_test = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)


from sklearn.preprocessing import StandardScaler, scale
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape,y_train.shape) #(354, 13) torch.Size([354, 1])
#<- scaler를 통과 하면 numpy가 된다

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)

from torch.utils.data import TensorDataset, DataLoader
train_set = TensorDataset(x_train, y_train) #x,y 합친다
test_set = TensorDataset(x_test, y_test)

# print(len(train_set))
# print(train_set[0])
'''
(354, 13) torch.Size([354, 1])
354
'''
train_loader = DataLoader(train_set, batch_size = 36, shuffle=True)
test_loader = DataLoader(test_set, batch_size = 36, shuffle=False)


class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        # super().__init__()
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_dim, 32)
        self.linear2 = nn.Linear(32, 32)
        self.linear3 = nn.Linear(32, 16)
        self.linear4 = nn.Linear(16, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        
    def forward(self, input_size):
        x = self.linear1(input_size)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        x = self.sigmoid(x)
        return x
    
model = Model(13,1).to(DEVICE) 

#모델
# model = nn.Sequential(
#     nn.Linear(13,32),
#     nn.ReLU(),
#     nn.Linear(32,32),
#     nn.ReLU(),
#     # nn.Linear(32,32),
#     # nn.ReLU(),
#     nn.Linear(32,16),
#     nn.ReLU(),
#     nn.Linear(16,1),
#     # nn.Sigmoid()
# ).to(DEVICE)
#컴파일, 훈현
criterion  = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.01)

def train(model, crirerion, optimizer, loader):
    total_loss = 0
    for x_batch, y_batch in loader:
        optimizer.zero_grad()    
        hypothesis = model(x_batch)
        loss = crirerion(hypothesis, y_batch)   
                
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(loader)

EPOCHS = 1000
for epoch in range (1, EPOCHS+1):
    loss = train(model, criterion, optimizer, train_loader)
    print('Epoch : {} / {}, Loss : {:.8f}'.format(epoch, EPOCHS, loss))
    

print('----#평가 예측----')    
#평가 예측
def evaluate(model,criterion, loader):
    model.eval()#평가모드  
    total_loss = 0
    
    for x_batch, y_batch in loader:      
        with torch.no_grad():
            predict = model(x_batch)
            loss = criterion(predict,y_batch)
            total_loss += loss.item()
    
    return total_loss     
    
loss = evaluate(model, criterion, test_loader)
print("최종 loss",loss)

y_predict = model(x_test)

from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

score = r2_score(y_test.cpu().numpy(),y_predict.cpu().detach().numpy())
print("r2_score : {:.4f}".format(score))

    

        