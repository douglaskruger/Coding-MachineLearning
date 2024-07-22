P = -1:.1:1;
T = [-.9602 -.5770 -.0729  .3771  .6405  .6600  .4609 ...
      .1336 -.2013 -.4344 -.5000 -.3930 -.1647  .0988 ...
      .3072  .3960  .3449  .1816 -.0312 -.2189 -.3201];
plot(P,T,'+');
 
title('Training Vectors');
xlabel('Input Vector P');
ylabel('Target Vector T');
p = -3:.1:3;
a = radbas(p);
plot(p,a)
title('Radial Basis Transfer Function');
xlabel('Input p');
ylabel('Output a');
 
a2 = radbas(p-1.5);
a3 = radbas(p+2);
a4 = a + a2*1 + a3*0.5;
plot(p,a,'b-',p,a2,'b--',p,a3,'b--',p,a4,'m-')
title('Weighted Sum of Radial Basis Transfer Functions');
xlabel('Input p');
ylabel('Output a');
 
eg = 0.02; % sum-squared error goal
sc = 1;    % spread constant
net = newrb(P,T,eg,sc);
 
plot(P,T,'+');
xlabel('Input');
 
X = -1:.01:1;
Y = sim(net,X);
 
hold on;
plot(X,Y);
hold off;
legend({'Target','Output'})