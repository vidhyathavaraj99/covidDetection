function [u, b, C]= level_set_line(u0,Img, b, K_sigma,KONE, nu,timestep,mu,epsilon, iter_lse)

u=u0; 
NB1 = conv2(b,K_sigma,'same');
NB2 = conv2(b.^2,K_sigma,'same');
C =updateC(Img, u, NB1, NB2, epsilon);

BOUND_Img = Img.^2.*KONE;
u = updateLSF(Img,u, C, BOUND_Img, NB1, NB2, mu, nu, timestep, epsilon, iter_lse);

Hu=Heaviside(u,epsilon);
M(:,:,1)=Hu;
M(:,:,2)=1-Hu;
b =updateB(Img, C, M,  K_sigma);


% update level set function
function u = updateLSF(Img, u0, C, KONE_Img, KB1, KB2, mu, nu, timestep, epsilon, iter_lse)
u=u0;
Hu=Heaviside(u,epsilon);
M(:,:,1)=Hu;
M(:,:,2)=1-Hu;
N_class=size(M,3);
e=zeros(size(M));
u=u0;
for kk=1:N_class
    e(:,:,kk) = KONE_Img - 2*Img.*C(kk).*KB1 + C(kk)^2*KB2;
end

for kk=1:iter_lse
    u=NeumannBoundCond(u);
    K=curvature_central(u);    % div()
    DiracU=Dirac(u,epsilon);
    ImageTerm=-DiracU.*(e(:,:,1)-e(:,:,2));
    penalizeTerm=mu*(4*del2(u)-K);
    lengthTerm=nu.*DiracU.*K;
    u=u+timestep*(lengthTerm+penalizeTerm+ImageTerm);
end

% update b
function b =updateB(Img, C, M,  Ksigma)

PC1=zeros(size(Img));
PC2=PC1;
N_class=size(M,3);
for kk=1:N_class
    PC1=PC1+C(kk)*M(:,:,kk);
    PC2=PC2+C(kk)^2*M(:,:,kk);
end
KNm1 = conv2(PC1.*Img,Ksigma,'same');
KDn1 = conv2(PC2,Ksigma,'same');

b = KNm1./KDn1;

% Update C
function C_new =updateC(Img, u, Kb1, Kb2, epsilon)
Hu=Heaviside(u,epsilon);
M(:,:,1)=Hu;
M(:,:,2)=1-Hu;
N_class=size(M,3);
for kk=1:N_class
    Nm2 = Kb1.*Img.*M(:,:,kk);
    Dn2 = Kb2.*M(:,:,kk);
    C_new(kk) = sum(Nm2(:))/sum(Dn2(:));
end



% Make a function  Neumann boundary condition
function g = NeumannBoundCond(f)
[nrow,ncol] = size(f);
g = f;
g([1 nrow],[1 ncol]) = g([3 nrow-2],[3 ncol-2]);  
g([1 nrow],2:end-1) = g([3 nrow-2],2:end-1);          
g(2:end-1,[1 ncol]) = g(2:end-1,[3 ncol-2]);  

function k = curvature_central(u)
% compute curvature for u with central difference scheme
[ux,uy] = gradient(u);
normDu = sqrt(ux.^2+uy.^2+1e-10);
Nx = ux./normDu;
Ny = uy./normDu;
[nxx,junk] = gradient(Nx);
[junk,nyy] = gradient(Ny);
k = nxx+nyy;

function h = Heaviside(x,epsilon)    
h=0.5*(1+(2/pi)*atan(x./epsilon));

function f = Dirac(x, epsilon)    
f=(epsilon/pi)./(epsilon^2.+x.^2);

