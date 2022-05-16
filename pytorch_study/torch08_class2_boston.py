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

def train(model, crirerion, optimizer, x_train, y_train):
    optimizer.zero_grad()    
    hypothesis = model(x_train)
    loss = crirerion(hypothesis,y_train) 
    
    loss.backward()
    optimizer.step()
    
    return loss.item()

EPOCHS = 1000
for epoch in range (1, EPOCHS+1):
    loss = train(model, criterion, optimizer, x_train, y_train)
    print('Epoch : {:4d} / {:4d}, Loss : {:.8f}'.format(epoch, EPOCHS, loss))
    

print('----#평가 예측----')    
#평가 예측

def evaluate(model,criterion,x_test,y_test):
    model.eval()#평가모드    
    with torch.no_grad():
        predict = model(x_test)
        loss = criterion(predict,y_test)
        return loss.item()    
    
loss = evaluate(model, criterion, x_test, y_test)
print("최종 loss",loss)

# y_predict = (model(x_test) >= 0.5).float()
y_predict = model(x_test)

# score = (y_predict == y_test).float().mean()
# print("accuracy : {:.4f}".format(score))


from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
# score = accuracy_score(y_predict.cpu().detach().numpy(),y_test.cpu().detach().numpy())
# print("accuracy : {:.4f}".format(score))

score = r2_score(y_test.cpu().numpy(),y_predict.cpu().detach().numpy())
print("r2_score : {:.4f}".format(score))

    

        