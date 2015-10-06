function  generate_scattorch_filters(inplanes, scale)

%in this script we generate the conjugate mirror filters to run on torch
options.maxorder=2;
options.J=scale;
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

current_order = zeros(inplanes,1);
nstates(1)=inplanes;
l0=2;
lpatten=l0*sqrt(2);

for j=1:options.J

rast=1;
for r=1:nstates(j)
	if current_order(r) < options.maxorder
		for l=1:options.L
		if j==1
		  W(r,:,:,rast) = real(filters.g0{l});rast=rast+1;
		  W(r,:,:,rast) = imag(filters.g0{l});
		else
		  W(r,:,:,rast) = real(filters.g{l});rast=rast+1;
		  W(r,:,:,rast) = imag(filters.g{l});
		end
		  new_order(round(rast/2)) = current_order(r)+1;rast=rast+1;
		end
	end
	if j==1
	W(r,:,:,rast) = filters.h0/lpatten; rast=rast+1;
	W(r,:,:,rast) = filters.h0/lpatten; 
	else
	W(r,:,:,rast) = filters.h/lpatten; rast=rast+1;
	W(r,:,:,rast) = filters.h/lpatten; 
	end
	new_order(round(rast/2)) = current_order(r);rast=rast+1;
end
nstates(j+1) = (rast -1)/2;
width(j) = size(filters.h,1);
downs(j) = 1 + (j>1);
%aux = reshape(W,2*nstates(j+1), nstates(j)*width(j)*width(j));
eval(['weights',num2str(j),'=','permute(W,[2,3,1,4]);']);
current_order = new_order;
clear W;
end

%generate the lowpass filters 
for j=1:options.J
rast=1;
for r=1:nstates(1)
if j==1
W(r, :, :, r) = filters.h0/l0;
else
W(r, :, :, r) = filters.h/l0;
end
end
eval(['lpweights',num2str(j),'=','permute(W,[2,3,1,4]);']);
clear W;
end


   %self.info = torch.load('/misc/vlgscratch2/LecunGroup/bruna/scattorch/wavelets_inplanes_' .. nInputPlane .. '_scale_' .. scale .. '.th' )
matfile = sprintf('/misc/vlgscratch2/LecunGroup/bruna/scattorch/wavelets_inplanes_%d_scale_%d.mat',inplanes, scale);
torchfile = sprintf('/misc/vlgscratch2/LecunGroup/bruna/scattorch/wavelets_inplanes_%d_scale_%d.th',inplanes, scale);


if scale==4
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

unix(sprintf('th /home/bruna/projects/scattorch/translate_scattfilters.lua -input %s -output %s',matfile, torchfile));
  

