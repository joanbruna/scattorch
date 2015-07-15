local scatteringMM, parent = torch.class('nn.scatteringMM', 'nn.Module')

function scatteringMM:__init(nInputPlane, scale)
   parent.__init(self)
   

   self.nInputPlane = nInputPlane

   self.padding = padding or 0
   self.scale = scale

	print(scale)

   self.info = torch.load('/misc/vlgscratch2/LecunGroup/bruna/scattorch/wavelets_inplanes_' .. nInputPlane .. '_scale_' .. scale .. '.th' )

   self.scatt = nn.Sequential()

   for i=1,self.scale do
      self.scatt:add(nn.SpatialConvolutionMM(self.info.nstates[i], 2*self.info.nstates[i+1], self.info.width[i], self.info.width[i], self.info.downs[i], self.info.downs[i]))
      self.scatt:add(nn.FeaturePooling(2,2))
   end
   --add normalization/whitening 
   --self.scatt:add(nn.SpatialConvolutionMM(self.info.nstates[self.scale+1], self.info.noutputs,1,1))

   for i=1,self.scale do
      self.scatt.modules[1+2*(i-1)].weight = self.info.weights[i]:clone()
      self.scatt.modules[1+2*(i-1)].bias:fill(0)
   end 
   
end

function scatteringMM:updateOutput(input)
      self.output = self.scatt:updateOutput(input)
      return self.output
end

function scatteringMM:updateGradInput(input, gradOutput)
    self.gradInput = self.scatt:updateGradInput(input, gradOutput)
    return self.gradInput
end


