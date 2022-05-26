clear all
close all

data_PMID=dlmread('~/Dropbox/LED-DM-EarlyUniverse/PPPC4DMID/AtProduction_gammas.dat','',1,0);
NS_PMID=179;
NP_PMID=size(data_PMID,1)/NS_PMID;
lgx=data_PMID(1:NS_PMID,2);
Eprim=data_PMID((linspace(1,NP_PMID,NP_PMID)-1)*NS_PMID+1);
%order: e, mu, tau, q, c, b, t, W, Z, gluon, gamma, h, nue, numu, nutau
dsub_PMID=data_PMID(:,[3,6,9,10,11,12,13,16,19,20,21,22,23,24,25]+2);
dntau=dsub_PMID(:,3);
%rows: secondary energies, columns: primary energies
dntaumatrix=reshape(dntau,[NS_PMID,NP_PMID]);
[lgEs,lgxs]=meshgrid(log10(Eprim),lgx);
lgdn=log10(dntaumatrix);
lgdn(isinf(lgdn))=-100;
mtau=1.777;
%2d extrapolation doesn't work well
%dns=10.^interp2(lgEs,lgxs,lgdn,ones(size(lgx))*log10(mtau),lgx,'makima');
%dns(isinf(dns))=0;
Es=logspace(log10(0.1),log10(5),20);
dnext=zeros(NS_PMID,length(Es));
for i=1:NS_PMID
    dnext(i,:)=10.^interp1(log10(Eprim),lgdn(i,:),log10(Es),'linear','extrap');
    dnext(i,Es<mtau)=0;
end
dnext(dnext<1e-50)=0;


figure
loglog(Eprim,dntaumatrix(1,:),'-+k');
hold on;
loglog(Eprim,dntaumatrix(10,:),'-+r');
loglog(Eprim,dntaumatrix(100,:),'-+b');


figure
loglog(10.^lgx,dntaumatrix(:,5),'-k')
hold on;
loglog(10.^lgx,dntaumatrix(:,1),'-r')
loglog(10.^lgx,dnext(:,18),'-b');
loglog(10.^lgx,dnext(:,15),'-m');

ylim([1e-8,1e1]);
xlabel('x=E_\gamma/E_\tau');
ylabel('dN/dlog_{10}x');
legend(['E_\tau=',num2str(Eprim(5)),'GeV'],['E_\tau=',num2str(Eprim(1)),'GeV'],...
    ['E_\tau=',num2str(Es(18)),'GeV'],['E_\tau=',num2str(Es(15)),'GeV'],'location','southeast','box','off');


data_PMID=dlmread('~/Dropbox/LED-DM-EarlyUniverse/PPPC4DMID/AtProduction_positrons.dat','',1,0);
%order: e, mu, tau, q, c, b, t, W, Z, gluon, gamma, h, nue, numu, nutau
dsub_PMID=data_PMID(:,[3,6,9,10,11,12,13,16,19,20,21,22,23,24,25]+2);
dntau=dsub_PMID(:,3);
%rows: secondary energies, columns: primary energies
dntaumatrix=reshape(dntau,[NS_PMID,NP_PMID]);
lgdn=log10(dntaumatrix);
lgdn(isinf(lgdn))=-100;
Es=logspace(log10(0.1),log10(5),20);
dnext_e=zeros(NS_PMID,length(Es));
for i=1:NS_PMID
    dnext_e(i,:)=10.^interp1(log10(Eprim),lgdn(i,:),log10(Es),'linear','extrap');
    dnext_e(i,Es<mtau)=0;
end
dnext_e(dnext_e<1e-50)=0;

figure
loglog(10.^lgx,dntaumatrix(:,5),'-k')
hold on;
loglog(10.^lgx,dntaumatrix(:,1),'-r')
loglog(10.^lgx,dnext_e(:,18),'-b');
loglog(10.^lgx,dnext_e(:,15),'-m');

ylim([1e-8,1e1]);
xlabel('x=E_e/E_\tau');
ylabel('dN/dlog_{10}x');
legend(['E_\tau=',num2str(Eprim(5)),'GeV'],['E_\tau=',num2str(Eprim(1)),'GeV'],...
    ['E_\tau=',num2str(Es(18)),'GeV'],['E_\tau=',num2str(Es(15)),'GeV'],'location','southeast','box','off');

%print output
dnext_e_out=dnext_e(:);
dnext_out=dnext(:);
lgx_out=repmat(lgx,[length(Es),1]);
Eprim_out=[];
for i=1:length(Es)
    Eprim_out=[Eprim_out;ones(NS_PMID,1)*Es(i)];
end

dout=[Eprim_out,lgx_out,dnext_e_out,dnext_out];
%save('secondariesfromtau.dat','dout','-ascii');





