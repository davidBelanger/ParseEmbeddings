a = dlmread('test.uz',' ');
a = a + 1;
aa = sptensor(aa,1);
rank = 10;
decomp = cp_als(aa,2);
%Z = elemfun(X, @sqrt)
child = decomp.U{1};
parent = decomp.U{2};
arc = decomp.U{3};
save('out',child,parent,arc);
