clear all
close all

me = 0.511e-3;
mmu = 0.10566;
mpiCh = 0.13957;
gamma=@(Eprim, mprim) Eprim/mprim;
beta=@(Eprim, mprim) sqrt(1.-1./gamma(Eprim,mprim).^2);
dGammadECM=@(Ee) sqrt(Ee.^2 - me^2).*(Ee.*(mmu^2 + me^2 - 2*mmu*Ee)...
    + 2*(Ee*mmu - me^2).*(mmu - Ee));
EeCM=@(Eelab, Eprim, mprim, y) gamma(Eprim, mprim).*Eelab.*(1-beta(Eprim,mprim).*y);
dNdEmu=@(Emu,Eelab) integral(@(y) 1./(2*gamma(Emu, mmu).*(beta(Emu,mmu).*y-1.)).*...
    dGammadECM(EeCM(Eelab,Emu,mmu,y)),...
    max([1./beta(Emu,mmu).*(1.-mmu./(2*gamma(Emu,mmu).*Eelab)), -1]),...
    min([1./beta(Emu,mmu).*(1.-me./(gamma(Emu,mmu).*Eelab)), 1]),'ArrayValued',true);
Eemin=@(Emu) max([me./gamma(Emu,mmu)./(1.+beta(Emu,mmu)),me]);
Eemax=@(Emu) min([mmu./(2*gamma(Emu,mmu).*(1.-beta(Emu,mmu))),Emu]);
% dNdEmu_norm=@(Emu) integral(@(Eelab) dNdEmu(Emu,Eelab),...
%     Eemin(Emu),Eemax(Emu));
dNdEmu_norm=@(Emu) quad(@(Eelab) dNdEmu(Emu,Eelab),...
    Eemin(Emu),Eemax(Emu));


Eprim=5;
Es=linspace(Eemin(Eprim),Eemax(Eprim),1000);
dndes=zeros(size(Es));
for i=1:length(dndes)
    dndes(i)=dNdEmu(Eprim,Es(i));
end
trapz(Es,dndes)
dNdEmu_norm(Eprim)


bet=@(gam) sqrt(1-1./gam.^2);
gams=logspace(0,3,1000);
As=1./(gams.^2.*(1-bet(gams)));
Bs=1./(gams.^2.*(1+bet(gams)));

