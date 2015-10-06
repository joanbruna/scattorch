
require 'torch'
require 'mattorch';
require 'xlua';
require 'nnx';

cmd = torch.CmdLine()
cmd:option('-input','M','input mat file')
cmd:option('-output','T', 'output torch file')

opt = cmd:parse(arg or {})

data = mattorch.load(opt.input)

scales= data.downs:size(1)

--we do it in a completely non-elegant way TODO repair
--
weights={}
lpweights={}

--print(data.lpweights1:size())
--print(data.weights1:size())

local ninputchannels = data.weights1:size(2)


if scales>0 then
	weights[1] = data.weights1:view(data.weights1:size(1),data.weights1:size(2)*data.weights1:size(3)*data.weights1:size(4))
	lpweights[1] = data.lpweights1:view(ninputchannels,-1)
if scales>1 then
	weights[2] = data.weights2:view(data.weights2:size(1),data.weights2:size(2)*data.weights2:size(3)*data.weights2:size(4))
	lpweights[2] = data.lpweights2:view(ninputchannels,-1)
if scales>2 then
	weights[3] = data.weights3:view(data.weights3:size(1),data.weights3:size(2)*data.weights3:size(3)*data.weights3:size(4))
	lpweights[3] = data.lpweights3:view(ninputchannels,-1)
if scales>3 then
	weights[4] = data.weights4:view(data.weights4:size(1),data.weights4:size(2)*data.weights4:size(3)*data.weights4:size(4))
	lpweights[4] = data.lpweights4:view(ninputchannels,-1)

end
end
end
end

downs={}
nstates={}
width={}

for i=1,scales do
downs[i]=data.downs[i][1]
nstates[i]=data.nstates[i][1]
width[i]=data.width[i][1]
end
nstates[scales+1]=data.nstates[scales+1][1]

torch.save(opt.output, {weights=weights, lpweights=lpweights, downs=downs, nstates=nstates, width=width } )

