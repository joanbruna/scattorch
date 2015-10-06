
require 'torch'
require 'mattorch';
require 'xlua';
require 'nnx';

cmd = torch.CmdLine()
cmd:option('-input','M','input mat file')
cmd:option('-output','T', 'output torch file')

opt = cmd:parse(arg or {})

data = mattorch.load(opt.input)

weights= data.weights:clone()
bias = data.bias:clone()
noutputs = bias:size(1)

torch.save(opt.output, {weights=weights, noutputs=noutputs, bias=bias } )

