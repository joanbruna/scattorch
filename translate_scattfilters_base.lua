
require 'torch'
require 'mattorch';
require 'xlua';
require 'nnx';

cmd = torch.CmdLine()
cmd:option('-input','M','input mat file')
cmd:option('-output','T', 'output torch file')

opt = cmd:parse(arg or {})

data = mattorch.load(opt.input)

--scales= data.downs:size(1)

weights={}
weights[1] = data.weights:view(data.weights:size(1),data.weights:size(2)*data.weights:size(3)*data.weights:size(4))

--downs=data.downs[1][1]
nstates=data.nstates[1][1]
width=data.width[1][1]

torch.save(opt.output, {weights=weights, nstates=nstates, width=width } )

