clear all;
close all;

addpath('~/BlackMax/');

d3=load('3d/gamma.txt');
d4=load('4d/gamma.txt');
d5=load('5d/gamma.txt');
d6=load('6d/gamma.txt');
daxes=dlmread('3d/axes.txt','',1,0);
lgE=daxes(:,1);
lgTBH=daxes(:,2);
[TT,EE]=meshgrid(lgTBH,lgE);

kn=@(n) (2^n*pi^((n-3)/2)*gamma((n+3)/2.)/(n+2))^(1/(n+1.));
gramtoGeV = 5.62e23;
MBH=@(T,n,Mstar) Mstar*((n+1.)/(4.*pi*kn(n))*Mstar/T_BH).^(n+1)/gramtoGeV;
mass=1e15;%g
M=mass*gramtoGeV;
n=3;
dNdE=d3/log(10)./(10.^lgE);%convert to dN/dEdt
%dNdE=d3;
Mstar=1e4;
r=kn(n)/Mstar*(M/Mstar).^(1./(n+1.));
T=(n+1.)/(4.*pi*r);
%dNint=interp2(TT,EE,dNdE,log10(ones(size(lgE))*T),lgE);
dNint=interp2(TT,EE,dNdE,-1*ones(size(lgE)),lgE);
loglog(10.^lgE,dNint);
% hold on;
% 
% n=6;
% dNdE=d6/log(10)./(10.^lgE);
% Mstar=1e4;
% r=kn(n)/Mstar*(M/Mstar).^(1./(n+1.));
% T=(n+1.)/(4.*pi*r);
% %dNint=interp2(TT,EE,dNdE,log10(ones(size(lgE))*T),lgE);
% dNint=interp2(TT,EE,dNdE,ones(size(lgE))*(-3),lgE);
% loglog(10.^lgE,dNint);

xlabel('E(GeV)');
ylabel('dN/dEdt');
%legend('n=3','n=6','location','northwest');

%printpdf(gcf,'testtable.pdf');


