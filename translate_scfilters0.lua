require 'torch'
require 'mattorch';
require 'xlua';
require 'nnx';

cmd = torch.CmdLine()
cmd:option('-input','M','input mat file')
cmd:option('-output','T', 'output torch file')

opt = cmd:parse(arg or {})

outfile='/misc/vlgscratch2/LecunGroup/bruna/scattorch/downsampling_filter.th'

data = mattorch.load('/misc/vlgscratch2/LecunGroup/bruna/scattorch/downsampling_filter.mat')

weights = data.downfilters
width= data.downfilters:size(1)

torch.save(outfile , {weights=weights, width=width } )

