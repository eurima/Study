from sklearn.datasets import load_breast_cancer
import torch
import torch.nn as nn
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print("torch :", torch.__version__, "DEVICE :",DEVICE)

#1.데이터
datasets = load_breast_cancer()

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

# print(x_train.shape,y_train.shape) #(398, 30) torch.Size([398, 1])
#<- scaler를 통과 하면 numpy가 된다

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)

#모델
model = nn.Sequential(
    nn.Linear(30,32),
    nn.ReLU(),
    nn.Linear(32,32),
    nn.ReLU(),
    # nn.Linear(32,32),
    # nn.ReLU(),
    nn.Linear(32,16),
    nn.ReLU(),
    nn.Linear(16,1),
    nn.Sigmoid()
).to(DEVICE)
#컴파일, 훈현
criterion  = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.01)

def train(model, crirerion, optimizer,x_train,y_train):
    optimizer.zero_grad()
    
    hypothesis = model(x_train)
    loss = crirerion(hypothesis,y_train) 
    
    loss.backward()
    optimizer.step()
    
    return loss.item()

EPOCHS = 1000
for epoch in range(1,EPOCHS + 1):
    loss = train(model, criterion, optimizer,x_train,y_train)
print("epoch : ",epoch, "loss : ",loss)
    

print('----#평가 예측----')    
#평가 예측

def evaluate(model,criterion,x_test,y_test):
    model.eval()#평가모드    
    with torch.no_grad():
        predict = model(x_test)
        loss2 = criterion(predict,y_test)
        return loss2.item()    
    
loss2 = evaluate(model, criterion, x_test, y_test)
print("최종 loss",loss2)

y_predict = (model(x_test) >= 0.5).float()

score = (y_predict == y_test).float().mean()
print("accuracy : {:.4f}".format(score))
# print(result.cpu().detach().numpy())

from sklearn.metrics import accuracy_score
score = accuracy_score(y_predict.cpu().detach().numpy(),y_test.cpu().detach().numpy())
print("accuracy : {:.4f}".format(score))

    

        