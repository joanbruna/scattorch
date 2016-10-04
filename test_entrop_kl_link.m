function test_entrop_kl_link(N)

M=10000;
epsi = 1e-2;
maxexpo = 0;
offs = 1;
expobase = 1;

for m=1:M
   %sample a pair of (close by) distributions 
   x = (rand(1,N)).^expobase;x = x/sum(x);
   y = x+rand*epsi*(rand(1,N).^(offs+rand*maxexpo)); y = y/sum(y);
   kl(m) = klsim(x,y);
   jsm(m) = jensen(x,y);
   tv(m) = tvdist(x,y);
   entro(m) = abs(entrop(x) - entrop(y));
end

close all
%figure(1)
%plot(kl, entro,'.')
figure(2)
plot(sqrt(kl), entro,'.')

figure(1)
plot(tv, entro, 'r.')
%figure(3)
%loglog(jsm, entro,'r.')


end

function out = tvdist(x,y)

    out = sum(abs(x-y));

end
function out = jensen(x,y)
    m = (x+y)/2;
    z1 = log(x./m);
    z2 = log(y./m);
    out=sum(x.*z1 + y.*z2);

end

function out = klsim(x,y)
    z = log(x./y);
    out=sum((x-y).*z);

end

function out = entrop(x)
    out=-sum(x.*log(x));
end