 
local TemporalFeaturePooling, parent = torch.class('nn.TemporalFeaturePooling', 'nn.Module')

function TemporalFeaturePooling:__init(kF,dF, pool_type ,eps)
   parent.__init(self)
   
   self.kF = kF or 2
   self.dF = dF or kF
   eps = eps or 0
   pool_type = pool_type or 'L2'

   if pool_type == 'L2' then
   	self.pooling_module = nn.SpatialLPPooling(1,2,kF,1,dF,1)
   	self.pooling_module:get(3).eps = eps
   elseif pool_type == 'Av' then
	self.pooling_module = nn.SpatialAveragePooling(1,2,kF,1,dF,1)
   end

end

function TemporalFeaturePooling:updateOutput(input)
   
   size_in = input:size()
   ndim = size_in:size()   
   if ndim == 3 then
	input_v = input:view( size_in[1], 1, size_in[2], size_in[3] )
	indp = 3
   else
	input_v = input:view( 1, size_in[1], size_in[2] )
	indp = 2
   end

   size_out = size_in
   --size_out[indp] = (size_in[indp] -1)/self.dF
   size_out[indp] = torch.floor(( size_in[indp]  - self.kF) /self.dF + 1) 

   self.output = self.pooling_module:forward(input_v):view( size_out)

   return self.output
end

function TemporalFeaturePooling:updateGradInput(input, gradOutput)
   
   size_in = input:size()
   size_g = gradOutput:size()
   ndim = size_in:size()
   if ndim == 3 then
        input_v = input:view( size_in[1],1, size_in[2], size_in[3]  )
	ind_p = 3
        gradOutput_v = gradOutput:view( size_g[1],1, size_g[2],size_g[3] )
   else
        input_v = input:view( 1, size_in[1], size_in[2] )
        gradOutput_v = gradOutput:view( 1, size_g[1], size_g[2] )
	ind_p = 2
   end


   self.gradInput = self.pooling_module:backward(input_v,gradOutput_v):view( size_in )	
   return self.gradInput
end


