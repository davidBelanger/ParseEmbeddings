a = dlmread('../test.small2',' ');
a = a + 1;
aa = sptensor(a,1);
clear a;
disp('loaded data');
rank = 100;
decomp = cp_als(aa,2);
%Z = elemfun(X, @sqrt)
child = decomp.U{1};
parent = decomp.U{2};
arc = decomp.U{3};
save('out','child','parent','arc');
