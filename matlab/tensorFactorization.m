function tensorFactorization(infile,rank)
a = dlmread(infile,' ');
a = a + 1;
aa = sptensor(a,1);
clear a;
disp('loaded data');
decomp = cp_als(aa,rank);
child = decomp.U{1};
parent = decomp.U{2};
arc = decomp.U{3};
save('out','child','parent','arc');
