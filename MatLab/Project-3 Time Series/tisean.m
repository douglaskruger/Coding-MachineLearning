% Tisean stuff
tiseanPath= 'C:\Tisean_3.0.0\bin\';

a=1.4;
system([tiseanPath,'henon -B0.3 -A',num2str(a),' -l1000 -o']);
x=load('henon.dat');

plot(x(:,1),x(:,2),'.')

%C:\project3>c:\Tisean_3.0.0\bin\mutual -l 1200 -x 1 -c 1 -b 10 -v 1 -d 10 -o output.txt Mackey-Glass.dat