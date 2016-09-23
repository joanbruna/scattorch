local scatt_1d_Downsampling, parent = torch.class('nn.scatt_1d_Downsampling', 'nn.Module')

function scatt_1d_Downsampling:__init(nInputPlane, path)
   parent.__init(self)
   
   self.nInputPlane = nInputPlane
   local pathf = path or '/misc/vlgscratch2/LecunGroup/bruna/scattorch/'

   self.padding = padding or 0
   self.scale = scale

   self.info = torch.load(pathf .. 'downsampling_1d_filter.th')

	---------------
	--main branch: scattering
	----------
   self.downs = nn.Sequential()
   self.downs:add(nn.SpatialConvolutionMM(nInputPlane, nInputPlane, self.info.width, 1, 2,1 ));
	local zz = self.info.width;
  	local ker = torch.Tensor(self.nInputPlane, zz*self.nInputPlane):zero()

	local k0flat = self.info.weights:clone():view(-1)

    for i=1, nInputPlane do
	ker:narrow(1,i,1):narrow(2,1+(i-1)*zz,zz):copy(k0flat)
    end

	self.downs.modules[1].weight = ker:clone()
	self.downs.modules[1].bias:fill(0)

end


function scatt_1d_Downsampling:updateOutput(input)
      self.output = self.downs:updateOutput(input)
      return self.output
end

function scatt_1d_Downsampling:updateGradInput(input, gradOutput)
    self.gradInput = self.downs:updateGradInput(input, gradOutput)
    return self.gradInput
end


