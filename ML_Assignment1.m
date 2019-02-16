clear all
ds = datastore('house_prices_data_training_data.csv','TreatAsMissing','NA',.....
    'MissingValue',0,'ReadSize',25000);
T = read(ds);
TrainingSet = T(1:10800,:); %70% of the data
CrossValidationSet = T(10801:14401,:);
TestSet = T(14402:17999,:);
size(T);
Alpha=.05;
;
m=length(T{:,1});
mTrain=length(TrainingSet{:,1});
mcv=length(CrossValidationSet{:,1});
mTest=length(TestSet{:,1});
%U0=T{:,2};
Utrain=TrainingSet{:,4:19};
Ucv=CrossValidationSet{:,4:19};
Utest=TestSet{:,4:19};
U1train=TrainingSet{:,20:21};
U1cv=CrossValidationSet{:,20:21};
U1test=TestSet{:,20:21};
U2train=TrainingSet{:,6:14};
U2cv=CrossValidationSet{:,6:14};
U2test=TestSet{:,6:14};
U3train=TrainingSet{:,13:21};
U3cv=CrossValidationSet{:,13:21};
U3test=TestSet{:,13:21};
U4train=TrainingSet{:,4:5};
U4cv=CrossValidationSet{:,4:5};
U4test=TestSet{:,4:5};
XTrain=[ones(mTrain,1) Utrain U1train U2train U3train U4train];%First hypothesis between U and U1
%Second hypothesis between U, U1 and U2
%third hypothesis between U, U1, U2 and U3
%fourth hypothesis between U, U1, U2, U3 and U4
Xcv=[ones(mcv,1) Ucv U1cv U2cv U3cv U4cv];
Xtest=[ones(mTest, 1) Utest U1test U2test U3test U4test];
nTrain=length(XTrain(1,:));
ncv=length(Xcv(1,:));
nTest=length(Xtest(1,:));
for w=2:nTrain
    if max(abs(XTrain(:,w)))~=0
    XTrain(:,w)=(XTrain(:,w)-mean((XTrain(:,w))))./std(XTrain(:,w));
    end
end

for q=2:ncv
    if max(abs(Xcv(:,q)))~=0
    Xcv(:,q)=(Xcv(:,q)-mean((Xcv(:,q))))./std(Xcv(:,q));
    end
end

for p=2:nTest
    if max(abs(Xtest(:,p)))~=0
    Xtest(:,p)=(Xtest(:,p)-mean((Xtest(:,p))))./std(Xtest(:,p));
    end
end

Ytrain=TrainingSet{:,3}/mean(TrainingSet{:,3});
Ycv=CrossValidationSet{:,3}/mean(CrossValidationSet{:,3});
Ytest=TestSet{:,3}/mean(TestSet{:,3});
Theta=zeros(nTrain,1);
k=1;

Etrain(k)=(1/(2*mTrain))*sum((XTrain*Theta-Ytrain).^2);
Rtrain=1;
while Rtrain==1
Alpha=Alpha*1;
Theta=Theta-(Alpha/mTrain)*XTrain'*(XTrain*Theta-Ytrain);
k=k+1
Etrain(k)=(1/(2*mTrain))*sum((XTrain*Theta-Ytrain).^2);
if Etrain(k-1)-Etrain(k)<0
    break
end 
qTrain=(Etrain(k-1)-Etrain(k))./Etrain(k-1);
if qTrain <.0001;
    Rtrain=0;
end
end
o=1;
Ecv(o)=(1/(2*mcv))*sum((Xcv*Theta-Ycv).^2);
Rcv=1;
while Rcv==1
o=o+1;
Ecv(o)=(1/(2*mcv))*sum((Xcv*Theta-Ycv).^2);
if Ecv(o-1)-Ecv(o)<0
    break
end
qcv=(Ecv(o-1)-Ecv(o))./Ecv(o-1);
if qcv <.0001;
    Rcv=0;
end
end
ErrorDiffWithFirstHypothesis=-0.0094;
ErrorDiffWithSecondHypothesis=-0.009;
ErrorDiffWithThirdHypothesis=-0.0089;
ErrorDiffWithFourthHypothesis=-0.0088;
%choosing the fourth hypothesis

g=1;
Etest(g)=(1/(2*mTest))*sum((Xtest*Theta-Ytest).^2);
Rtest=1;
while Rtest==1
    g=g+1;
    Etest(g)=(1/(2*mTest))*sum((Xtest*Theta-Ytest).^2);
if Etest(g-1)-Etest(g)<0
    break
end
qTest=(Etest(g-1)-Etest(g))./Etest(g-1);
if qTest <.0001;
    Rtest=0;
end
end











