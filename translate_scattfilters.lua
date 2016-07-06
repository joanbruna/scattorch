
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
if scales>4 then
	weights[5] = data.weights5:view(data.weights5:size(1),data.weights5:size(2)*data.weights5:size(3)*data.weights5:size(4))
	lpweights[5] = data.lpweights5:view(ninputchannels,-1)
if scales>5 then
	weights[6] = data.weights6:view(data.weights6:size(1),data.weights6:size(2)*data.weights6:size(3)*data.weights6:size(4))
	lpweights[6] = data.lpweights6:view(ninputchannels,-1)
if scales>6 then
	weights[7] = data.weights7:view(data.weights7:size(1),data.weights7:size(2)*data.weights7:size(3)*data.weights7:size(4))
	lpweights[7] = data.lpweights7:view(ninputchannels,-1)

end
end
end
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

