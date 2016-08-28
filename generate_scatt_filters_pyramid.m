function filters=generate_scatt_filters_pyramid(options)

L = getoptions(options,'L',8);

sigma=.8;%2/sqrt(3); 
sigma0=.8;%2*sigma/3;
sigma1 = sqrt(2*sigma^2-sigma0^2);
sigma2 = sqrt(1.5)*sigma0;
xi = 2*pi/3;
xi1 = xi*3/4;% * 2 * sigma^2/sigma1^2

optfilt.size_filter=[9 9];
optfilt.L=L;
optfilt.sigma_phi=sigma0; 
optfilt.sigma_psi=sigma;
optfilt.xi_psi=xi;
fi=morlet_filter_bank_2d_pyramid(optfilt);
filters.h0= fi.h.filter.coefft;
for l=1:size(fi.g.filter,2)
filters.g0{l}=fi.g.filter{l}.coefft;
%conjugate filters
filters.conjg0{l} = conj(flipud(fliplr(filters.g0{l}))); 
end

optfilt.L=L;
optfilt.sigma_phi=sigma2;
optfilt.sigma_psi=sigma1;
optfilt.xi_psi=xi1;
fi=morlet_filter_bank_2d_pyramid(optfilt);
filters.h= fi.h.filter.coefft;
for l=1:size(fi.g.filter,2)
filters.g{l}=fi.g.filter{l}.coefft;
%conjugate filters
filters.conjg{l} = conj(flipud(fliplr(filters.g{l}))); 
end

if options.rototranslation
filters.roto = compact_roto_matrix(options);
end

