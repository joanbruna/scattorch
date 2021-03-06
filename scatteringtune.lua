local scatteringtuneMM, parent = torch.class('nn.scatteringtuneMM', 'nn.Module')

function scatteringtuneMM:__init(nInputPlane, scale)
   parent.__init(self)
   
   self.nInputPlane = nInputPlane

   self.padding = padding or 0
   self.scale = scale

   self.info = torch.load('/misc/vlgscratch2/LecunGroup/bruna/scattorch/wavelets_inplanes_' .. nInputPlane .. '_scale_' .. scale .. '.th' )

	---------------
	--main branch: scattering
	----------
   self.scatt = nn.Sequential()

	-- attention: I am not doing any smoothing after the modulus. the filters are not exactly the same TODO check the impact 
	-- of this simplification 
   for i=1,self.scale do
      self.scatt:add(nn.SpatialConvolutionMM(self.info.nstates[i], 2*self.info.nstates[i+1], self.info.width[i], self.info.width[i], self.info.downs[i], self.info.downs[i]))
      self.scatt:add(nn.FeaturePooling(2,2))
   end
   --add normalization by a simple constant factor (TODO improve)
   self.scatt:add(nn.Mul())

   for i=1,self.scale do
      self.scatt.modules[1+2*(i-1)].weight = self.info.weights[i]:clone()
      self.scatt.modules[1+2*(i-1)].bias:fill(0)
   end 
	local scalingfact = torch.Tensor(1):fill(2^(2*self.scale-1))
	self.scatt.modules[1+2*self.scale].weight = scalingfact;   
   
	----------------
	-- add the lowpass in a separate branch
	-- ---------------
  	self.lpass = nn.Sequential()
  	for i=1,self.scale do
	self.lpass:add(nn.SpatialConvolutionMM(self.info.nstates[1], self.info.nstates[1], self.info.width[i], self.info.width[i], self.info.downs[i], self.info.downs[i]))
	end
	self.lpass:add(nn.Mul())
  	for i=1,self.scale do
	self.lpass.modules[i].weight = self.info.lpweights[i]:clone()
	self.lpass.modules[i].bias:fill(0)
	end
	self.lpass.modules[self.scale+1].weight = scalingfact;

	---- -------
	-- add the TV branch (finest Haar scale)
	-- -----------
	self.haar = nn.Sequential()
	self.haar:add(nn.SpatialConvolutionMM(self.info.nstates[1], 2*self.info.nstates[1], self.info.width[1], self.info.width[1]))
	self.haar:add(nn.FeaturePooling(2,2))
	for i=2,self.scale do
		self.haar:add(nn.SpatialConvolutionMM(self.info.nstates[1], self.info.nstates[1], self.info.width[i], self.info.width[i], self.info.downs[i], self.info.downs[i]))
	end
	self.haar:add(nn.Mul())

	--define the haar filters implementing the TV
	local zz = self.info.width[1]*self.info.width[1]
  	local ker1 = torch.Tensor(2*self.nInputPlane, zz*self.info.nstates[1]):zero()

	for i=1,self.info.nstates[1] do
		ker1[i][zz*(i-1)+(zz-1)/2+1]=1
		ker1[i][zz*(i-1)+(zz-1)/2+2]=-1
		ker1[self.nInputPlane+i][zz*(i-1)+(zz-1)/2+1]=1
		ker1[self.nInputPlane+i][zz*(i-1)+(zz-1)/2+1+self.info.width[1]]=-1
	end
	self.haar.modules[1].weight = ker1:clone()
	self.haar.modules[1].bias:fill(0)
	for i=2,self.scale do
		self.haar.modules[i+1].weight = self.info.lpweights[i]:clone()
		self.haar.modules[i+1].bias:fill(0)
	end
	self.haar.modules[self.scale+2].weight = scalingfact;

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

local function makeContiguous(self, input, gradOutput)
   if not input:isContiguous() then
      self._input = self._input or input.new()
      self._input:resizeAs(input):copy(input)
      input = self._input
   end
   if gradOutput then
      if not gradOutput:isContiguous() then
     self._gradOutput = self._gradOutput or gradOutput.new()
     self._gradOutput:resizeAs(gradOutput):copy(gradOutput)
     gradOutput = self._gradOutput
      end
   end
   return input, gradOutput
end

function scatteringtuneMM:updateOutput(input)
      input = makeContiguous(self, input)
      self.output = self.all:updateOutput(input)
      return self.output
end

function scatteringtuneMM:updateGradInput(input, gradOutput)
   if self.gradInput then
      input, gradOutput = makeContiguous(self, input, gradOutput)
    self.gradInput = self.all:updateGradInput(input, gradOutput)
    return self.gradInput
   end
end

function scatteringtuneMM:accGradParameters(input, gradOutput, scale)
   input, gradOutput = makeContiguous(self, input, gradOutput)
   return self.all:accGradParameters(self, input, gradOutput, scale)
end
