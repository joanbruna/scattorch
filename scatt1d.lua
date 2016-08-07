--require 'AMul'
--require 'scattDownsampling'
--require 'FeaturePooling'

local scatt1d, parent = torch.class('nn.scatt1d', 'nn.Module')

function scatt1d:__init(nInputPlane, scale)
   parent.__init(self)
   
   self.nInputPlane = nInputPlane

   self.padding = padding or 0
   self.scale = scale
   self.pad = 0

   self.info = torch.load('/misc/vlgscratch2/LecunGroup/bruna/scattorch/wavelets_inplanes_1d_' .. nInputPlane .. '_scale_' .. scale .. '.th' )

	---------------
	--main branch: scattering
	----------
   self.scatt = nn.Sequential()
	local scalingfact = 2^(2*self.scale-1)

	-- attention: I am not doing any smoothing after the modulus. the filters are not exactly the same TODO check the impact 
	-- of this simplification 
   for i=1,self.scale do
      self.scatt:add(nn.TemporalConvolution(self.info.nstates[i], 2*self.info.nstates[i+1], self.info.width[i]))
      self.scatt:add(nn.FeaturePooling(2,2))
      if i > 1 then
	self.scatt:add(nn.scatt_1d_Downsampling(self.info.nstates[i+1]))
      end
   end
   --add normalization by a simple constant factor (TODO improve)
   self.scatt:add(nn.AMul(scalingfact))

	self.scatt.modules[1].weight = self.info.weights[1]:clone()
	self.scatt.modules[1].bias:fill(0)

   for i=2,self.scale do
      self.scatt.modules[3*(i-1)].weight = self.info.weights[i]:clone()
      self.scatt.modules[3*(i-1)].bias:fill(0)
   end 
   
	----------------
	-- add the lowpass in a separate branch
	-- ---------------
  	self.lpass = nn.Sequential()
  	for i=1,self.scale do
	self.lpass:add(nn.TemporalConvolution(self.info.nstates[1], self.info.nstates[1], self.info.width[i]))
	if i >1 then
	self.lpass:add(nn.scatt_1d_Downsampling(self.info.nstates[1]))
	end
	end
	self.lpass:add(nn.AMul(scalingfact))
	self.lpass.modules[1].weight = self.info.lpweights[1]:clone()
	self.lpass.modules[1].bias:fill(0)
  	for i=2,self.scale do
	self.lpass.modules[2*(i-1)].weight = self.info.lpweights[i]:clone()
	self.lpass.modules[2*(i-1)].bias:fill(0)
	end

	---- -------
	-- add the TV branch (finest Haar scale)
	-- -----------
	self.haar = nn.Sequential()
	self.haar:add(nn.TemporalConvolution(self.info.nstates[1], 2*self.info.nstates[1], self.info.width[1]))
	self.haar:add(nn.FeaturePooling(2,2))
	for i=2,self.scale do
		self.haar:add(nn.TemporalConvolution(self.info.nstates[1], self.info.nstates[1], self.info.width[i]))
		self.haar:add(nn.scatt_1d_Downsampling(self.info.nstates[1]))
	end
	self.haar:add(nn.AMul(scalingfact))

	--define the haar filters implementing the TV
	local zz = self.info.width[1]
  	local ker1 = torch.Tensor(2*self.nInputPlane, zz*self.info.nstates[1]):zero()

	for i=1,self.info.nstates[1] do
		ker1[i][zz*(i-1)+(zz-1)/2+1]=1
		ker1[i][zz*(i-1)+(zz-1)/2+2]=-1
		ker1[self.nInputPlane+i][zz*(i-1)+(zz-1)/2+1]=1
		ker1[self.nInputPlane+i][zz*(i-1)+(zz-1)/2+2]=-1
	end
	self.haar.modules[1].weight = ker1:clone()
	self.haar.modules[1].bias:fill(0)
	for i=2,self.scale do
		self.haar.modules[2*i-1].weight = self.info.lpweights[i]:clone()
		self.haar.modules[2*i-1].bias:fill(0)
	end

	--------------------------
	--join everything together
	-----------------------
	self.joint = nn.ConcatTable()
	self.joint:add(self.scatt)
	self.joint:add(self.lpass)
	self.joint:add(self.haar)

	self.all = nn.Sequential()
	self.all:add(self.joint)
	self.all:add(nn.JoinTable(1,3))

end

function scatt1d:updateOutput(input)
      self.output = self.all:updateOutput(input)
      return self.output
end

function scatt1d:updateGradInput(input, gradOutput)
    self.gradInput = self.all:updateGradInput(input, gradOutput)
    return self.gradInput
end


