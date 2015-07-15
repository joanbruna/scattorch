function  generate_scattorch_filters_single

%in this script we generate the conjugate mirror filters to run on torch
options.maxorder=2;
options.J=4;
options.L=8;
options.Jroto=3;
options.rototranslation=0;
options.incpu=1;
options.pad = 0;
options.precision='double';

options

filters=generate_scatt_filters_pyramid(options);

%we need to write the fields
%	weights
%	nstates
%	width
%	downs

%there is one filter-bank per scale. all the paths are regrouped into a single convolutional tensor. 

nstates=options.L+1;
lpatten=10*sqrt(2);
%if alias
for l=1:options.L
	W(1,:,:,2*l-1) = real(filters.g0{l});
	W(1,:,:,2*l) = imag(filters.g0{l});
end
	W(1,:,:,2*options.L+1) = filters.h0/lpatten;
	W(1,:,:,2*options.L+2) = filters.h0/lpatten; 
width = size(filters.h,1);
downs = 1;
weights = permute(W, [2, 3, 1,4]);

matfile = '/misc/vlgscratch2/LecunGroup/bruna/scattorch/wavelets_base0.mat';
torchfile = '/misc/vlgscratch2/LecunGroup/bruna/scattorch/wavelets_base0.th';
save(matfile,'weights','nstates','width','downs','-v7.3');
unix(sprintf('th /home/bruna/projects/inv/th/translate_scattfilters_base.lua -input %s -output %s',matfile, torchfile));

%else

for l=1:options.L
	W(1,:,:,2*l-1) = real(filters.g{l});
	W(1,:,:,2*l) = imag(filters.g{l});
end
	W(1,:,:,2*options.L+1) = filters.h/lpatten;
	W(1,:,:,2*options.L+2) = filters.h/lpatten; 
width = size(filters.h,1);
downs = 2;
weights = permute(W, [2, 3, 1,4]);
matfile = '/misc/vlgscratch2/LecunGroup/bruna/scattorch/wavelets_base1.mat';
torchfile = '/misc/vlgscratch2/LecunGroup/bruna/scattorch/wavelets_base1.th';
save(matfile,'weights','nstates','width','downs','-v7.3');
unix(sprintf('th /home/bruna/projects/inv/th/translate_scattfilters_base.lua -input %s -output %s',matfile, torchfile));
  


