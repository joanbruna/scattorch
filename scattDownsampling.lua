local scattDownsampling, parent = torch.class('nn.scattDownsampling', 'nn.Module')

function scattDownsampling:__init(nInputPlane)
   parent.__init(self)
   
   self.nInputPlane = nInputPlane

   self.padding = padding or 0
   self.scale = scale

   self.info = torch.load('/misc/vlgscratch2/LecunGroup/bruna/scattorch/downsampling_filter.th')

	---------------
	--main branch: scattering
	----------
   self.downs = nn.Sequential()
   self.downs:add(nn.SpatialConvolutionMM(nInputPlane, nInputPlane, self.info.width, self.info.width, 2, 2, 1, 1));

--  self.downs.modules[1].weight = torch.CudaTensor(nInputPlane, nInputPlance*self.info.width*self.info.width):zero();
	local zz = self.info.width*self.info.width;
  	local ker = torch.Tensor(self.nInputPlane, zz*self.nInputPlane):zero()

	local k0flat = self.info.weights:clone():view(-1)

    for i=1, nInputPlane do
	ker:narrow(1,i,1):narrow(2,1+(i-1)*zz,zz):copy(k0flat)
    end

	self.downs.modules[1].weight = ker:clone()
	self.downs.modules[1].bias:fill(0)

end


function scattDownsampling:updateOutput(input)
      self.output = self.downs:updateOutput(input)
      return self.output
end

function scattDownsampling:updateGradInput(input, gradOutput)
    self.gradInput = self.downs:updateGradInput(input, gradOutput)
    return self.gradInput
end


