n=10^3;
%n=10^5;
%n=10^7;

a=1;
b=1/(2^0.5);

x=unifrnd(0,1,n,1);
y=unifrnd(0,1,n,1);

x=2*a.*x-a;
y=2*b.*y-b;

region=(1/(a*a)).*(x.*x)+(1/(b*b)).*(y.*y);
m=find(region<=1);

nm=length(m);
area=4*a*b*nm/n
