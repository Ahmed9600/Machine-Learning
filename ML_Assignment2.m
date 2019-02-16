clear all
ds = datastore('heart_DD.csv','TreatAsMissing','NA',.....
    'MissingValue',0,'ReadSize',25000);
T = read(ds);
TrainingSet = T(1:150,:); %60% of the data for training
CrossValidationSet = T(151:201,:); %20% for cross validation test
TestSet = T(202:250,:); %20% for the test set
size(T);
Alpha=.05;
;
m=length(T{:,1});
mTrain=length(TrainingSet{:,1});
mcv=length(CrossValidationSet{:,1});
mTest=length(TestSet{:,1});
Utrain=TrainingSet{:,12:14};
Ucv=CrossValidationSet{:,12:14};
Utest=TestSet{:,12:14};
U1train=TrainingSet{:,3:6};
U1cv=CrossValidationSet{:,3:6};
U1test=TestSet{:,3:6};
U2train=TrainingSet{:,7:9};
U2cv=CrossValidationSet{:,7:9};
U2test=TestSet{:,7:9};
U3train=TrainingSet{:,10:12};
U3cv=CrossValidationSet{:,10:12};
U3test=TestSet{:,10:12};
U4train=TrainingSet{:,1:3};
U4cv=CrossValidationSet{:,1:3};
U4test=TestSet{:,1:3};
XTrain=[ones(mTrain,1) Utrain U1train U2train U3train U4train];%First hypothesis between U and U1
%Second hypothesis between U, U1 and U2
%third hypothesis between U, U1, U2 and U3
%fourth hypothesis between U, U1, U2, U3 and U4
%Choosing the fourth hypothesis which gave the least error difference
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
Etrain(k)=(-1/(mTrain))*sum((Ytrain.*log(XTrain*Theta))+(1-Ytrain).*log(XTrain*Theta));
Rtrain=1;
while Rtrain==1
Alpha=Alpha*1;
Theta=Theta-(Alpha/mTrain)*XTrain'*(XTrain*Theta-Ytrain);
k=k+1
Etrain(k)=(-1/(mTrain))*sum((Ytrain.*log(XTrain*Theta))+(1-Ytrain).*log(XTrain*Theta));
if Etrain(k-1)-Etrain(k)<0
    break
end 
qTrain=(Etrain(k-1)-Etrain(k))./Etrain(k-1);
if qTrain <.0001;
    Rtrain=0;
end
end
o=1;
Ecv(o)=(-1/(mcv))*sum((Ycv.*log(Xcv*Theta))+(1-Ycv).*log(Xcv*Theta));
Rcv=1;
while Rcv==1
o=o+1;
Ecv(o)=(-1/(mcv))*sum((Ycv.*log(Xcv*Theta))+(1-Ycv).*log(Xcv*Theta));
if Ecv(o-1)-Ecv(o)<0
    break
end
qcv=(Ecv(o-1)-Ecv(o))./Ecv(o-1);
if qcv <.0001;
    Rcv=0;
end
end

g=1;
Etest(g)=(-1/(mTest))*sum((Ytest.*log(Xtest*Theta))+(1-Ytest).*log(Xtest*Theta));
Rtest=1;
while Rtest==1
    g=g+1;
    Etest(g)=(-1/(mTest))*sum((Ytest.*log(Xtest*Theta))+(1-Ytest).*log(Xtest*Theta));
if Etest(g-1)-Etest(g)<0
    break
end
qTest=(Etest(g-1)-Etest(g))./Etest(g-1);
if qTest <.0001;
    Rtest=0;
end
end











