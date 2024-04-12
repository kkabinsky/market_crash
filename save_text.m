

x=wha;

%load advanc.txt
f=fopen('wha_python.txt','w');
 l=size(x)
 k=l(1,1)
%       fprintf(fid,'%6.2f  %12.8f\n',y);
for i=1:k
    fprintf(f,'%2.6f,',x(i))
end
fclose(f);