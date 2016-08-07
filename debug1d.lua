require 'nn';
require 'TemporalFeaturePooling.lua';
require 'AMul.lua';
require 'scatt_1d_Downsampling.lua';
require 'scatt1d'

bla = torch.rand(10,10000,1)

scale = 9
nInputPlane = 1
info = torch.load('/misc/vlgscratch2/LecunGroup/bruna/scattorch/wavelets_1d_inplanes_' .. nInputPlane .. '_scale_' .. scale .. '.th' )


   scatt = nn.Sequential()
   for i=1,scale do
      scatt:add(nn.TemporalConvolution(info.nstates[i], 2*info.nstates[i+1], info.width[i]))
      scatt:add(nn.TemporalFeaturePooling(2,2))
      if i > 1 then
	scatt:add(nn.scatt_1d_Downsampling(info.nstates[i+1]))
      end
   end

  lpass = nn.Sequential()
  for i=1,scale do
	lpass:add(nn.TemporalConvolution(info.nstates[1], info.nstates[1], info.width[i]))
	if i >1 then
		lpass:add(nn.scatt_1d_Downsampling(info.nstates[1]))
	end
  end

   haar = nn.Sequential()
   haar:add(nn.TemporalConvolution(info.nstates[1], 2*info.nstates[1], info.width[1]))
   haar:add(nn.TemporalFeaturePooling(2,2))
	for i=2,scale do
		haar:add(nn.TemporalConvolution(info.nstates[1], info.nstates[1], info.width[i]))
		haar:add(nn.scatt_1d_Downsampling(info.nstates[1]))
	end

print(bla:size())
bli = scatt:forward(bla)
print(bli:size())
ble = lpass:forward(bla)
print(ble:size())
blo = haar:forward(bla)
print(blo:size())

