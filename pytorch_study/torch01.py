
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f

#----1. 데이터 정제하여 투입
x = np.array([1,2,3])
y = np.array([1,2,3])

# x = torch.FloatTensor(x)
# y = torch.FloatTensor(y)
# print(x,y)
# print(x.shape, y.shape)
'''
tensor([1., 2., 3.]) tensor([1., 2., 3.])
torch.Size([3]) torch.Size([3])  스칼라3개짜리 벡터 하나!

pytorch는 x, y 를 행렬로 바꿔야 한다
(3,) -> (3,1) 행렬 상태로 바꿔야 한다
'''
x = torch.FloatTensor(x).unsqueeze(1)
y = torch.FloatTensor(y).unsqueeze(1)
#(3,) -> (3,1)

#----2. 모델 구현
# model = Sequential() #Sequential 클래스의 인스턴스
# model.add(Dense(1, input_dim = 1)) #출력 1, 인풋1 ( 1단 구조 )
model = nn.Linear(1,1) #인풋, 아웃풋 즉 (3,1)-> 행무시 열

#----3. 컴파일 , 훈련
#model.compile(loss='mse', optimizer='adam') 
criterion = nn.MSELoss(reduction='sum') #통상 loss 변수명은 criterion로 쓴다
optimizer = optim.Adam(model.parameters(),lr=0.01)

# print(optimizer)
'''
Adam (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    eps: 1e-08
    lr: 0.01
    weight_decay: 0
)
'''
def train(model,criterion,optimizer,x,y):
    # model.train() #훈련모드
    optimizer.zero_grad() #가중치 초기화
    hyperthesis = model(x)
    loss = criterion(hyperthesis,y) #순전파
    # loss = (hyperthesis - y).sum()
    loss.backward() #역전파 기울기값 계산
    optimizer.step()       #기울기 수정
    
    return loss.item()
    
epochs = 500
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


result = model(torch.Tensor([[4]]))
print("4의 예측값",result)
    

        

    

    

    
    
'''    
    

# mse : 최소자승법, adam 역할은 loss의 mse 감소 시키는 역할....일단은
model.fit(x, y, epochs=2000, batch_size=1) 
#epochs 훈련양, batch_size 몇개씩 넣을 것인가 --> 속도, 성능, 과적합 여부에 따라

#----4. 성과 예측
loss = model.evaluate(x,y)
print('loss : ', loss)
pre_x = 4
result = model.predict([pre_x])
print(f'{pre_x} 의 예측값은 : ',result[0][0])

2000  epoch
loss :  0.2507708966732025
4 의 예측값은 :  2.9459848
'''
#Git 테스트~///


