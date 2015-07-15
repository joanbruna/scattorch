local wavereshape, parent = torch.class('nn.wavereshape', 'nn.Module')

function wavereshape:__init(  )
   parent.__init(self)
end

function wavereshape:updateOutput(input)
   	size_in = input:size()
   	ndim = size_in:size()   

   	if ndim == 4 then
	input_v = input:view( size_in[1]*size_in[2], 1, size_in[3], size_in[4]  )
   	else
	input_v = input:view( size_in[1], 1, size_in[2], size_in[3]  )
   	end

      	return input_v
end

function waveletMM:updateGradInput(input, gradOutput)

	self.gradInput = gradOutput:view(input:size())

    	return self.gradInput
end


