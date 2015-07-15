local waveletMM, parent = torch.class('nn.waveletMM', 'nn.Module')

function waveletMM:__init( antialias )
   parent.__init(self)
   
   --self.nInputPlane = nInputPlane

   self.padding = padding or 0
   --self.scale = scale
   self.antialias = antialias or 1

--	print(scale)
    if self.antialias == 0 then
   self.info = torch.load('/misc/vlgscratch2/LecunGroup/bruna/scattorch/wavelets_base0.th')
    else
   self.info = torch.load('/misc/vlgscratch2/LecunGroup/bruna/scattorch/wavelets_base1.th')
    end

   self.wavel = nn.Sequential()
   self.K = self.info.nstates
  
   --self.wavel:add(nn.wavereshape())
   self.wavel:add(nn.SpatialConvolutionMM(1, 2*self.info.nstates, self.info.width, self.info.width, self.info.downs, self.info.downs))
   self.wavel:add(nn.FeaturePooling(2,2))

   self.wavel.modules[1].weight = self.info.weights[1]:clone()
   self.wavel.modules[1].bias:fill(0)
   
end

function waveletMM:updateOutput(input)
   	size_in = input:size()
   	ndim = size_in:size()   
	--print(ndim)
   	if ndim == 4 then
	input_v = input:view( size_in[1]*size_in[2], 1, size_in[3], size_in[4]  )
	indp = 2
   	else
	input_v = input:view( size_in[1], 1, size_in[2], size_in[3]  )
	indp = 1
   	end
   	size_out = size_in
   	size_out[indp] = size_in[indp]*self.info.nstates

        input_w = self.wavel:forward(input_v)
	--print(input:size())
	--print(input:size():size())
	--print(input_w:size())
	--print(ndim)
	--print(size_in)
	if input:size():size() == 4 then
	self.output = input_w:view(size_in[1], size_in[2]*(input_w:size(2)), input_w:size(3), input_w:size(4))
	else
        self.output = input_w:view((input_w:size(1))*(input_w:size(2)), input_w:size(3), input_w:size(4))
	end

      return self.output
end

function waveletMM:updateGradInput(input, gradOutput)

   	size_in = input:size()
	size_gin = gradOutput:size()
   	ndim = size_in:size()   
   	if ndim == 4 then
	input_v = input:view( size_in[1]*size_in[2], 1, size_in[3], size_in[4]  )
	input_g = gradOutput:view(size_in[1]*size_in[2], size_gin[2]/size_in[2], size_gin[3], size_gin[4])
   	else
	input_v = input:view( size_in[1], 1, size_in[2], size_in[3]  )
	input_g = gradOutput:view(size_in[1], size_gin[1]/size_in[1], size_gin[2], size_gin[3])
   	end
	--self.gradInput = self.wavel:backward(input_v, input_g):view( size_in )
	self.gradInput = self.wavel:backward(input_v, input_g):view( input:size())
    return self.gradInput
end


