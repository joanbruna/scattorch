local scatt1d, parent = torch.class('nn.scatt1d', 'nn.Module')

function scatt1d:__init(nInputPlane, scale, order, Q, path)

	parent.__init(self)

	self.nInputPlane = nInputPlane
	local pathf = path or '/misc/vlgscratch2/LecunGroup/bruna/scattorch/'

	self.padding = padding or 0
	self.scale = scale
	self.pad = 0
	self.order = order or 2
	self.Q = Q or 1

	if Q == 0 then
		self.info = torch.load(pathf .. 'wavelets_1d_inplanes_' .. nInputPlane .. '_scale_' .. scale .. '_maxorder_' .. self.order .. '.th' )
	else
		self.info = torch.load(pathf .. 'wavelets_1d_inplanes_' .. nInputPlane .. '_scale_' .. scale .. '_maxorder_' .. self.order .. '_Q' .. self.Q .. '.th' )
	end

	---------------
	--main branch: scattering
	----------
	self.scatt = nn.Sequential()
	local scalingfact = 2^(self.scale-1)

	for i=1,self.scale do
		self.scatt:add(nn.SpatialConvolutionMM(self.info.nstates[i], 2*self.info.nstates[i+1], self.info.width[i], 1))
		self.scatt:add(nn.FeaturePooling(2,2))
		if i > 1 then
			self.scatt:add(nn.scatt_1d_Downsampling(self.info.nstates[i+1], pathf))
		end
	end

	self.scatt.modules[1].weight:copy(self.info.weights[1])
	self.scatt.modules[1].bias:fill(0)

	for i=2,self.scale do
	self.scatt.modules[3*(i-1)].weight:copy(self.info.weights[i])
	self.scatt.modules[3*(i-1)].bias:fill(0)
	end 

	----------------
	-- add the lowpass in a separate branch
	-- ---------------
	self.lpass = nn.Sequential()
	for i=1,self.scale do
		self.lpass:add(nn.SpatialConvolutionMM(self.info.nstates[1], self.info.nstates[1], self.info.width[i],1))
		if i >1 then
		self.lpass:add(nn.scatt_1d_Downsampling(self.info.nstates[1], pathf))
		end
	end

	self.lpass.modules[1].weight:copy(self.info.lpweights[1])
	self.lpass.modules[1].bias:fill(0)
	for i=2,self.scale do
		self.lpass.modules[2*(i-1)].weight:copy(self.info.lpweights[i])
		self.lpass.modules[2*(i-1)].bias:fill(0)
	end

	---- -------
	-- add the TV branch (finest Haar scale)
	-- -----------
	self.haar = nn.Sequential()
	self.haar:add(nn.SpatialConvolutionMM(self.info.nstates[1], 2*self.info.nstates[1], self.info.width[1],1))
	self.haar:add(nn.FeaturePooling(2,2))
	for i=2,self.scale do
		self.haar:add(nn.SpatialConvolutionMM(self.info.nstates[1], self.info.nstates[1], self.info.width[i],1))
		self.haar:add(nn.scatt_1d_Downsampling(self.info.nstates[1], pathf))
	end

	--define the haar filters implementing the TV
	local zz = self.info.width[1]
	local ker1 = torch.Tensor(2*self.nInputPlane, zz*self.info.nstates[1]):zero()

	for i=1,self.info.nstates[1] do
		ker1[i][zz*(i-1)+(zz-1)/2+1]=1/2
		ker1[i][zz*(i-1)+(zz-1)/2+2]=-1/2
		ker1[self.nInputPlane+i][zz*(i-1)+(zz-1)/2+1]=1/2
		ker1[self.nInputPlane+i][zz*(i-1)+(zz-1)/2+2]=-1/2
	end
	self.haar.modules[1].weight:copy(ker1)
	self.haar.modules[1].bias:fill(0)
	for i=2,self.scale do
		self.haar.modules[2*i-1].weight:copy(self.info.lpweights[i])
		self.haar.modules[2*i-1].bias:fill(0)
	end

	--------------------------
	--join everything together
	-----------------------
	self.joint = nn.ConcatTable()
	self.joint:add(self.scatt)
	self.joint:add(self.lpass)
	self.joint:add(self.haar)

	--self.all = nn.Sequential()
	--self.all:add(self.scatt)

	self.all = nn.Sequential()
	self.all:add(self.joint)
	self.all:add(nn.JoinTable(2,2))

end

function scatt1d:updateOutput(input)
	self.output = self.all:updateOutput(input)
	return self.output
end

function scatt1d:updateGradInput(input, gradOutput)
	self.gradInput = self.all:updateGradInput(input, gradOutput)
	return self.gradInput
end


