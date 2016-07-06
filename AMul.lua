local AMul, parent = torch.class('nn.AMul', 'nn.Module')

function AMul:__init(c)
   parent.__init(self)
	self.c = c
end

function AMul:updateOutput(input)
	self.output = input:mul(self.c)
      return self.output
end

function AMul:updateGradInput(input, gradOutput)
	self.gradInput=gradOutput:mul(self.c)
    return self.gradInput
end


