local haartv, parent = torch.class('nn.haartv', 'nn.Module')

function haartv:__init(nInputPlane)
   parent.__init(self)
   
   self.nInputPlane = nInputPlane

   
   self.haar = nn.Sequential()
   self.haar:add(nn.SpatialConvolutionMM(self.nInputPlane, 2*self.nInputPlane, 3, 3))
   self.haar:add(nn.FeaturePooling(2,2))

   local zz=9;
   local ker = torch.Tensor(2*self.nInputPlane, 9*self.nInputPlane):zero()

	for i=1,self.nInputPlane do
		ker[i][zz*(i-1)+(zz-1)/2+1]=1
		ker[i][zz*(i-1)+(zz-1)/2+2]=-1
		ker[nInputPlane+i][zz*(i-1)+(zz-1)/2+1]=1
		ker[nInputPlane+i][zz*(i-1)+(zz-1)/2+1+3]=-1
	end
	self.haar.modules[1].weight = ker:clone()
	self.haar.modules[1].bias:fill(0)

end

function haartv:updateOutput(input)
      self.output = self.haar:updateOutput(input)
      return self.output
end

function haartv:updateGradInput(input, gradOutput)
    self.gradInput = self.haar:updateGradInput(input, gradOutput)
    return self.gradInput
end



