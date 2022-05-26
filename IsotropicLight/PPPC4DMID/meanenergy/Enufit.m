clear all
close all

%Eprimary Enumean for electron muon tau quark charm bottom top w-boson z-boson 
%gluon gamma higgs nue numu nutau
data=dlmread('Enumean.txt','',1,0);
for i=2:size(data,2)
    loglog(data(:,1),data(:,i));
    hold on;
end
Eprim=log10(data(:,1));
%y=-0.1585*x^2+2.651*x-5.463
Ee=log10(data(:,2));
Eprime=Eprim(Ee>0.1);
Ee=Ee(Ee>0.1);
%y=0.9963*x-0.1984
Emuon=log10(data(:,3));
%y=0.9954*x-0.1661
Etau=log10(data(:,4));
%y=0.9998*x-0.3824
Equark=log10(data(:,5));
%y=1.002*x-0.3992
Echarm=log10(data(:,6));
%y=0.9951*x-0.3327
Ebottom=log10(data(:,7));
%y=0.9919*x-0.2891
Etop=log10(data(:,8));
%y=0.9965*x-0.2839
Ew=log10(data(:,9));
%y=1*x-0.2713
Ez=log10(data(:,10));
%y=0.9997*x-0.3954
Egluon=log10(data(:,11));
%y=-0.03768*x^3+0.3717*x^2+0.2567*x-1.752
Egamma=log10(data(:,12));
%y=1.011*x-0.3512
Ehiggs=log10(data(:,13));
%y=0.9733*x+0.005131
Enue=log10(data(:,14));
%y=0.9874*x-0.02277
Enumu=log10(data(:,15));
%y=0.9885*x-0.02476
Enutau=log10(data(:,16));




