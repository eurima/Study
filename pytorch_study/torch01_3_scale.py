import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print("torch :", torch.__version__, "DEVICE :",DEVICE)
#----1. 데이터 정제하여 투입
x = np.array([1,2,3])
y = np.array([1,2,3])

x = torch.FloatTensor(x)#.to(DEVICE)#.unsqueeze(1).to(DEVICE)
y = torch.FloatTensor(y).unsqueeze(1).to(DEVICE)

x = (x - torch.mean(x))/torch.std(x)

print(x)
x = torch.FloatTensor(x).unsqueeze(1).to(DEVICE)#.unsqueeze(1).to(DEVICE)



#----2. 모델 구현
# model = nn.Linear(1,1).to(DEVICE) #인풋, 아웃풋 즉 (3,1)-> 행무시 열
model = nn.Sequential(
    nn.Linear(1,5),
    nn.Linear(5,3),
    nn.Linear(3,4),
    nn.Linear(4,2),
    nn.Linear(2,1),    
).to(DEVICE)

#----3. 컴파일 , 훈련
criterion = nn.MSELoss() #통상 loss 변수명은 criterion로 쓴다
optimizer = optim.Adam(model.parameters(),lr=0.001)

def train(model,criterion,optimizer,x,y):
    # model.train() #훈련모드 default값임
    optimizer.zero_grad() #가중치 초기화
    hyperthesis = model(x)
    # loss = criterion(hyperthesis,y) #1번
    
    
    # loss = nn.MSELoss()(hyperthesis,y) #2번
    loss = F.mse_loss(hyperthesis,y) #3번
    loss.backward() #역전파 기울기값 계산
    optimizer.step()       #기울기 수정
    
    return loss.item()
    
epochs = 1000
for epoch in range(1,epochs+1):
    loss = train(model,criterion,optimizer,x,y)
    print(f"epoch : {epoch}, loss : {loss}")
    
#평가 예측
def evaluate(model,criterion,x,y):
    model.eval()#평가모드
    
    with torch.no_grad():
        predict = model(x)
        loss2 = criterion(predict,y)
        return loss2.item()    
    
loss2 = evaluate(model,criterion,x,y)
print("최종 loss",loss2)

result = model(torch.Tensor([[4]]).to(DEVICE))#.to(DEVICE)
print("4의 예측값",result)

