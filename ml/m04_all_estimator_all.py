from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score, r2_score
import numpy as np

from sklearn.datasets import load_iris, load_wine, load_diabetes, load_boston, load_breast_cancer
import warnings
warnings.filterwarnings(action='ignore')

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

datasets = {'Iris':load_iris(),'Wine':load_wine(),'Diabets':load_diabetes(),'Cancer':load_breast_cancer(),'Boston':load_boston()}
for (dataset_name,datasets) in datasets.items():
    dataset = load_iris()
    alAlgorithms_classifier = all_estimators(type_filter='classifier')#classifier'41 regressor 55, transformer 79, cluster 10
    alAlgorithms_regressor = all_estimators(type_filter='regressor')
    num = 0

    x = dataset.data
    y = dataset.target

    x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size = 0.8, shuffle = True, random_state = 66)

    scaler = MinMaxScaler()
    scaler.fit_transform(x_train)
    scaler.transform(x_test)
    
    # print(len(np.unique(y_train)))
    
    if len(np.unique(y_train)) <2:
    
        print('--alAlgorithms_classifier',dataset_name)
        print('=================================================')   
    
        for (name, algo) in alAlgorithms_classifier:
            try:
                model = algo()
                model.fit(x_train,y_train)
                y_predict = model.predict(x_test)
                acc = accuracy_score(y_test, y_predict)
                print(name, '의 정답률 : ',acc)

            except:
                #print(name,' Err')
                num = num + 1
        print('---------------------------------')
        # print('모델의 갯수 : ',len(alAlgorithms_classifier))
        #print('classifier 안되는 모델의 갯수 : ',num)
    
    else:
        print('--alAlgorithms_regressor',dataset_name)    
        for (name, algo) in alAlgorithms_regressor:
            try:
                model = algo()
                model.fit(x_train, y_train)
                y_predict = model.predict(x_test)
                acc = r2_score(y_test, y_predict)
                print(name, '의 정답률 : ', acc)

            except:
                #print(name, ' Err')
                num = num + 1
        print('---------------------------------')
    # print('모델의 갯수 : ',len(alAlgorithms_classifier))
    #print('regressor 안되는 모델의 갯수 : ', num)

'''
--alAlgorithms_regressor Iris
ARDRegression 의 정답률 :  0.9018396130429636
AdaBoostRegressor 의 정답률 :  0.9304170688448374
BaggingRegressor 의 정답률 :  0.9587248322147651 
BayesianRidge 의 정답률 :  0.9055436990297966    
CCA 의 정답률 :  0.8187077323940747
DecisionTreeRegressor 의 정답률 :  0.8993288590604027
DummyRegressor 의 정답률 :  -0.010486577181207712    
ElasticNet 의 정답률 :  0.7066108603068253
ElasticNetCV 의 정답률 :  0.9054184007985565
ExtraTreeRegressor 의 정답률 :  0.8993288590604027
ExtraTreesRegressor 의 정답률 :  0.946236577181208
GaussianProcessRegressor 의 정답률 :  0.6236633942581697
GradientBoostingRegressor 의 정답률 :  0.9350320415701814
HistGradientBoostingRegressor 의 정답률 :  0.928390605664707
HuberRegressor 의 정답률 :  0.9036707892477875
KNeighborsRegressor 의 정답률 :  0.9375838926174497
KernelRidge 의 정답률 :  0.9063477491227561
Lars 의 정답률 :  0.9044795932990088
LarsCV 의 정답률 :  0.9044795932990088
Lasso 의 정답률 :  0.41069764012321464
LassoCV 의 정답률 :  0.9049931850279125
LassoLars 의 정답률 :  -0.010486577181207712
LassoLarsCV 의 정답률 :  0.9044795932990088
LassoLarsIC 의 정답률 :  0.8896320119539592
LinearRegression 의 정답률 :  0.9044795932990096
LinearSVR 의 정답률 :  0.9067565365869628
MLPRegressor 의 정답률 :  0.9075375011224036
NuSVR 의 정답률 :  0.9244673733335868
OrthogonalMatchingPursuit 의 정답률 :  0.8837955385010984
OrthogonalMatchingPursuitCV 의 정답률 :  0.9044795932990088
PLSCanonical 의 정답률 :  0.257250414805852
PLSRegression 의 정답률 :  0.8894807187718152
PassiveAggressiveRegressor 의 정답률 :  0.9037120955084428
PoissonRegressor 의 정답률 :  0.808700291055618
QuantileRegressor 의 정답률 :  -0.00671140940044368
RANSACRegressor 의 정답률 :  0.9029668326723158
RadiusNeighborsRegressor 의 정답률 :  0.9242164261428097
RandomForestRegressor 의 정답률 :  0.9550553691275168
Ridge 의 정답률 :  0.9059577552890712
RidgeCV 의 정답률 :  0.9048357192276848
SGDRegressor 의 정답률 :  0.8825333401341758
SVR 의 정답률 :  0.9213943776064587
TheilSenRegressor 의 정답률 :  0.8924574286985707
TransformedTargetRegressor 의 정답률 :  0.9044795932990096
TweedieRegressor 의 정답률 :  0.8513555448837072
---------------------------------
--alAlgorithms_regressor Wine
ARDRegression 의 정답률 :  0.9018396130429636
AdaBoostRegressor 의 정답률 :  0.9064859911485289
BaggingRegressor 의 정답률 :  0.9557046979865772
BayesianRidge 의 정답률 :  0.9055436990297966
CCA 의 정답률 :  0.8187077323940747
DecisionTreeRegressor 의 정답률 :  0.8993288590604027
DummyRegressor 의 정답률 :  -0.010486577181207712
ElasticNet 의 정답률 :  0.7066108603068253
ElasticNetCV 의 정답률 :  0.9054184007985565
ExtraTreeRegressor 의 정답률 :  0.7483221476510067
ExtraTreesRegressor 의 정답률 :  0.9345234899328859
GaussianProcessRegressor 의 정답률 :  0.6236633942581697
GradientBoostingRegressor 의 정답률 :  0.9347058263924826
HistGradientBoostingRegressor 의 정답률 :  0.928390605664707
HuberRegressor 의 정답률 :  0.9036707892477875
KNeighborsRegressor 의 정답률 :  0.9375838926174497
KernelRidge 의 정답률 :  0.9063477491227561
Lars 의 정답률 :  0.9044795932990088
LarsCV 의 정답률 :  0.9044795932990088
Lasso 의 정답률 :  0.41069764012321464
LassoCV 의 정답률 :  0.9049931850279125
LassoLars 의 정답률 :  -0.010486577181207712
LassoLarsCV 의 정답률 :  0.9044795932990088
LassoLarsIC 의 정답률 :  0.8896320119539592
LinearRegression 의 정답률 :  0.9044795932990096
LinearSVR 의 정답률 :  0.9024963545790153
MLPRegressor 의 정답률 :  0.898535071061335
NuSVR 의 정답률 :  0.9244673733335868
OrthogonalMatchingPursuit 의 정답률 :  0.8837955385010984
OrthogonalMatchingPursuitCV 의 정답률 :  0.9044795932990088
PLSCanonical 의 정답률 :  0.257250414805852
PLSRegression 의 정답률 :  0.8894807187718152
PassiveAggressiveRegressor 의 정답률 :  0.902174893827503
PoissonRegressor 의 정답률 :  0.808700291055618
QuantileRegressor 의 정답률 :  -0.00671140940044368
RANSACRegressor 의 정답률 :  0.9044795932990096
RadiusNeighborsRegressor 의 정답률 :  0.9242164261428097
RandomForestRegressor 의 정답률 :  0.9578338926174497
Ridge 의 정답률 :  0.9059577552890712
RidgeCV 의 정답률 :  0.9048357192276848
SGDRegressor 의 정답률 :  0.882091421255601
SVR 의 정답률 :  0.9213943776064587
TheilSenRegressor 의 정답률 :  0.8899387628311135
TransformedTargetRegressor 의 정답률 :  0.9044795932990096
TweedieRegressor 의 정답률 :  0.8513555448837072
---------------------------------
--alAlgorithms_regressor Diabets
ARDRegression 의 정답률 :  0.9018396130429636
AdaBoostRegressor 의 정답률 :  0.8960249147850887
BaggingRegressor 의 정답률 :  0.964261744966443
BayesianRidge 의 정답률 :  0.9055436990297966
CCA 의 정답률 :  0.8187077323940747
DecisionTreeRegressor 의 정답률 :  0.9496644295302014
DummyRegressor 의 정답률 :  -0.010486577181207712
ElasticNet 의 정답률 :  0.7066108603068253
ElasticNetCV 의 정답률 :  0.9054184007985565
ExtraTreeRegressor 의 정답률 :  0.7483221476510067
ExtraTreesRegressor 의 정답률 :  0.9360184563758389
GaussianProcessRegressor 의 정답률 :  0.6236633942581697
GradientBoostingRegressor 의 정답률 :  0.9341969256049033
HistGradientBoostingRegressor 의 정답률 :  0.928390605664707
HuberRegressor 의 정답률 :  0.9036707892477875
KNeighborsRegressor 의 정답률 :  0.9375838926174497
KernelRidge 의 정답률 :  0.9063477491227561
Lars 의 정답률 :  0.9044795932990088
LarsCV 의 정답률 :  0.9044795932990088
Lasso 의 정답률 :  0.41069764012321464
LassoCV 의 정답률 :  0.9049931850279125
LassoLars 의 정답률 :  -0.010486577181207712
LassoLarsCV 의 정답률 :  0.9044795932990088
LassoLarsIC 의 정답률 :  0.8896320119539592
LinearRegression 의 정답률 :  0.9044795932990096
LinearSVR 의 정답률 :  0.8988171533209317
MLPRegressor 의 정답률 :  0.8852790161305348
NuSVR 의 정답률 :  0.9244673733335868
OrthogonalMatchingPursuit 의 정답률 :  0.8837955385010984
OrthogonalMatchingPursuitCV 의 정답률 :  0.9044795932990088
PLSCanonical 의 정답률 :  0.257250414805852
PLSRegression 의 정답률 :  0.8894807187718152
PassiveAggressiveRegressor 의 정답률 :  0.890996801964615
PoissonRegressor 의 정답률 :  0.808700291055618
QuantileRegressor 의 정답률 :  -0.00671140940044368
RANSACRegressor 의 정답률 :  0.9005778978505612
RadiusNeighborsRegressor 의 정답률 :  0.9242164261428097
RandomForestRegressor 의 정답률 :  0.9588003355704698
Ridge 의 정답률 :  0.9059577552890712
RidgeCV 의 정답률 :  0.9048357192276848
SGDRegressor 의 정답률 :  0.8868521892393153
SVR 의 정답률 :  0.9213943776064587
TheilSenRegressor 의 정답률 :  0.8913499104885282
TransformedTargetRegressor 의 정답률 :  0.9044795932990096
TweedieRegressor 의 정답률 :  0.8513555448837072
---------------------------------
--alAlgorithms_regressor Cancer
ARDRegression 의 정답률 :  0.9018396130429636
AdaBoostRegressor 의 정답률 :  0.9147727515889325
BaggingRegressor 의 정답률 :  0.9385906040268457
BayesianRidge 의 정답률 :  0.9055436990297966
CCA 의 정답률 :  0.8187077323940747
DecisionTreeRegressor 의 정답률 :  0.9496644295302014
DummyRegressor 의 정답률 :  -0.010486577181207712
ElasticNet 의 정답률 :  0.7066108603068253
ElasticNetCV 의 정답률 :  0.9054184007985565
ExtraTreeRegressor 의 정답률 :  0.8489932885906041
ExtraTreesRegressor 의 정답률 :  0.9503288590604027
GaussianProcessRegressor 의 정답률 :  0.6236633942581697
GradientBoostingRegressor 의 정답률 :  0.9340856526000871
HistGradientBoostingRegressor 의 정답률 :  0.928390605664707
HuberRegressor 의 정답률 :  0.9036707892477875
KNeighborsRegressor 의 정답률 :  0.9375838926174497
KernelRidge 의 정답률 :  0.9063477491227561
Lars 의 정답률 :  0.9044795932990088
LarsCV 의 정답률 :  0.9044795932990088
Lasso 의 정답률 :  0.41069764012321464
LassoCV 의 정답률 :  0.9049931850279125
LassoLars 의 정답률 :  -0.010486577181207712
LassoLarsCV 의 정답률 :  0.9044795932990088
LassoLarsIC 의 정답률 :  0.8896320119539592
LinearRegression 의 정답률 :  0.9044795932990096
LinearSVR 의 정답률 :  0.9061911980726052
MLPRegressor 의 정답률 :  0.8827684116251112
NuSVR 의 정답률 :  0.9244673733335868
OrthogonalMatchingPursuit 의 정답률 :  0.8837955385010984
OrthogonalMatchingPursuitCV 의 정답률 :  0.9044795932990088
PLSCanonical 의 정답률 :  0.257250414805852
PLSRegression 의 정답률 :  0.8894807187718152
PassiveAggressiveRegressor 의 정답률 :  0.8870962676997006
PoissonRegressor 의 정답률 :  0.808700291055618
QuantileRegressor 의 정답률 :  -0.00671140940044368
RANSACRegressor 의 정답률 :  0.9044795932990096
RadiusNeighborsRegressor 의 정답률 :  0.9242164261428097
RandomForestRegressor 의 정답률 :  0.9553221476510068
Ridge 의 정답률 :  0.9059577552890712
RidgeCV 의 정답률 :  0.9048357192276848
SGDRegressor 의 정답률 :  0.8541484919073359
SVR 의 정답률 :  0.9213943776064587
TheilSenRegressor 의 정답률 :  0.8920759728568869
TransformedTargetRegressor 의 정답률 :  0.9044795932990096
TweedieRegressor 의 정답률 :  0.8513555448837072
---------------------------------
--alAlgorithms_regressor Boston
ARDRegression 의 정답률 :  0.9018396130429636
AdaBoostRegressor 의 정답률 :  0.9018762071515478
BaggingRegressor 의 정답률 :  0.9441275167785235
BayesianRidge 의 정답률 :  0.9055436990297966
CCA 의 정답률 :  0.8187077323940747
DecisionTreeRegressor 의 정답률 :  0.9496644295302014
DummyRegressor 의 정답률 :  -0.010486577181207712
ElasticNet 의 정답률 :  0.7066108603068253
ElasticNetCV 의 정답률 :  0.9054184007985565
ExtraTreeRegressor 의 정답률 :  0.8993288590604027
ExtraTreesRegressor 의 정답률 :  0.941993288590604
GaussianProcessRegressor 의 정답률 :  0.6236633942581697
GradientBoostingRegressor 의 정답률 :  0.933391954099099
HistGradientBoostingRegressor 의 정답률 :  0.928390605664707
HuberRegressor 의 정답률 :  0.9036707892477875
KNeighborsRegressor 의 정답률 :  0.9375838926174497
KernelRidge 의 정답률 :  0.9063477491227561
Lars 의 정답률 :  0.9044795932990088
LarsCV 의 정답률 :  0.9044795932990088
Lasso 의 정답률 :  0.41069764012321464
LassoCV 의 정답률 :  0.9049931850279125
LassoLars 의 정답률 :  -0.010486577181207712
LassoLarsCV 의 정답률 :  0.9044795932990088
LassoLarsIC 의 정답률 :  0.8896320119539592
LinearRegression 의 정답률 :  0.9044795932990096
LinearSVR 의 정답률 :  0.9045658138908605
MLPRegressor 의 정답률 :  0.49967542507429785
NuSVR 의 정답률 :  0.9244673733335868
OrthogonalMatchingPursuit 의 정답률 :  0.8837955385010984
OrthogonalMatchingPursuitCV 의 정답률 :  0.9044795932990088
PLSCanonical 의 정답률 :  0.257250414805852
PLSRegression 의 정답률 :  0.8894807187718152
PassiveAggressiveRegressor 의 정답률 :  0.8656926796898763
PoissonRegressor 의 정답률 :  0.808700291055618
QuantileRegressor 의 정답률 :  -0.00671140940044368
RANSACRegressor 의 정답률 :  0.9044795932990096
RadiusNeighborsRegressor 의 정답률 :  0.9242164261428097
RandomForestRegressor 의 정답률 :  0.9580755033557047
Ridge 의 정답률 :  0.9059577552890712
RidgeCV 의 정답률 :  0.9048357192276848
SGDRegressor 의 정답률 :  0.8866840038768639
SVR 의 정답률 :  0.9213943776064587
TheilSenRegressor 의 정답률 :  0.8914537689503857
TransformedTargetRegressor 의 정답률 :  0.9044795932990096
TweedieRegressor 의 정답률 :  0.8513555448837072
---------------------------------


'''