function  generate_scattorch_1d_filters(inplanes, scale, maxorder)

%in this script we generate the conjugate mirror filters to run on torch
options.maxorder=maxorder;
options.J=scale;
options.incpu=1;
options.precision='double';
options

addpath('/home/bruna/matlab/toolbox/synthesis/rotosynthesis');

% get filters
%[Faf, Fsf] = FSfarras;
%[af, sf] = dualfilt1;

filters = morlet_1d_pyramid(options);

%we need to write the fields
%	weights
%	nstates
%	width
%	downs

%there is one filter-bank per scale. all the paths are regrouped into a single convolutional tensor. 


current_order = zeros(inplanes,1);
nstates(1)=inplanes;
l0=1;
lpatten=l0*sqrt(2);

for j=1:options.J

rast=1;
for r=1:nstates(j)
	if current_order(r) < options.maxorder
		if j==1
		  W(r,:,rast) = real(filters.g0);rast=rast+1;
		  W(r,:,rast) = imag(filters.g0);
		else
		  W(r,:,rast) = real(filters.g);rast=rast+1;
		  W(r,:,rast) = imag(filters.g);
		end
		  new_order(round(rast/2)) = current_order(r)+1;rast=rast+1;
	end
	if j==1
	W(r,:,rast) = filters.h0/lpatten; rast=rast+1;
	W(r,:,rast) = filters.h0/lpatten; 
	else
	W(r,:,rast) = filters.identity/lpatten; rast=rast+1;
	W(r,:,rast) = filters.identity/lpatten; 
	end
	new_order(round(rast/2)) = current_order(r);rast=rast+1;
end
nstates(j+1) = (rast -1)/2;
width(j) = size(filters.h,1);
downs(j) = 1 + (j>1);
%aux = reshape(W,2*nstates(j+1), nstates(j)*width(j)*width(j));
%eval(['weights',num2str(j),'=','permute(W,[2,1,3]);']);
eval(['weights',num2str(j),'=','permute(W,[1,2,3]);']);
current_order = new_order;
clear W;
end

%generate the lowpass filters 
for j=1:options.J
rast=1;
for r=1:nstates(1)
if j==1
W(r, :, r) = filters.h0/l0;
else
W(r, :, r) = filters.identity/l0;
end
end
%eval(['lpweights',num2str(j),'=','permute(W,[2,1,3]);']);
eval(['lpweights',num2str(j),'=','permute(W,[1,2,3]);']);
clear W;
end

downfilters = filters.downfilters;
downfilters = 2*downfilters/sum(downfilters(:));
save('/misc/vlgscratch2/LecunGroup/bruna/scattorch/downsampling_1d_filter.mat','downfilters');


   %self.info = torch.load('/misc/vlgscratch2/LecunGroup/bruna/scattorch/wavelets_inplanes_' .. nInputPlane .. '_scale_' .. scale .. '.th' )
%matfile = sprintf('/misc/vlgscratch2/LecunGroup/bruna/scattorch/wavelets_1d_inplanes_%d_scale_%d.mat',inplanes, scale);
%torchfile = sprintf('/misc/vlgscratch2/LecunGroup/bruna/scattorch/wavelets_1d_inplanes_%d_scale_%d.th',inplanes, scale);
matfile = sprintf('/misc/vlgscratch2/LecunGroup/bruna/scattorch/wavelets_1d_inplanes_%d_scale_%d_maxorder_%d.mat',inplanes, scale, options.maxorder);
torchfile = sprintf('/misc/vlgscratch2/LecunGroup/bruna/scattorch/wavelets_1d_inplanes_%d_scale_%d_maxorder_%d.th',inplanes, scale, options.maxorder);

if scale==10
save(matfile,'lpweights1','lpweights2', 'lpweights3','lpweights4','lpweights5', 'lpweights6', 'lpweights7','lpweights8', 'lpweights9', 'lpweights10','weights1','weights2', 'weights3','weights4','weights5', 'weights6', 'weights7','weights8', 'weights9', 'weights10','nstates','width','downs','-v7.3');
elseif scale==9
save(matfile,'lpweights1','lpweights2', 'lpweights3','lpweights4','lpweights5', 'lpweights6', 'lpweights7','lpweights8', 'lpweights9', 'weights1','weights2', 'weights3','weights4','weights5', 'weights6', 'weights7','weights8', 'weights9', 'nstates','width','downs','-v7.3');
elseif scale==8
save(matfile,'lpweights1','lpweights2', 'lpweights3','lpweights4','lpweights5', 'lpweights6', 'lpweights7','lpweights8', 'weights1','weights2', 'weights3','weights4','weights5', 'weights6', 'weights7','weights8', 'nstates','width','downs','-v7.3');
elseif scale==7
save(matfile,'lpweights1','lpweights2', 'lpweights3','lpweights4','lpweights5', 'lpweights6', 'lpweights7','weights1','weights2', 'weights3','weights4','weights5', 'weights6', 'weights7','nstates','width','downs','-v7.3');
elseif scale==6
save(matfile,'lpweights1','lpweights2', 'lpweights3','lpweights4','lpweights5', 'lpweights6', 'weights1','weights2', 'weights3','weights4','weights5', 'weights6','nstates','width','downs','-v7.3');
elseif scale==5
save(matfile,'lpweights1','lpweights2', 'lpweights3','lpweights4','lpweights5','weights1','weights2', 'weights3','weights4','weights5','nstates','width','downs','-v7.3');
elseif scale==4
save(matfile,'lpweights1','lpweights2', 'lpweights3', 'lpweights4','weights1','weights2', 'weights3', 'weights4','nstates','width','downs','-v7.3');
elseif scale==3
save(matfile,'lpweights1','lpweights2', 'lpweights3','weights1','weights2', 'weights3', 'nstates','width','downs','-v7.3');

elseif scale==2
save(matfile,'lpweights1','lpweights2','weights1','weights2', 'nstates','width','downs','-v7.3');

elseif scale==1
save(matfile,'lpweights1','weights1','nstates','width','downs','-v7.3');

else
error('to do')
end

unix(sprintf('th /home/bruna/projects/scattorch/translate_scattfilters_1d.lua -input %s -output %s',matfile, torchfile));
  

