require 'randomkit'

local BinaryLinear, parent = torch.class('BinaryLinear', 'nn.Module')

function BinaryLinear:__init(inputSize, outputSize,stcWeights)
   parent.__init(self)

   self.weight = torch.Tensor(outputSize, inputSize)
   self.weightB = torch.Tensor(outputSize, inputSize)
   self.weightOrg = torch.Tensor(outputSize, inputSize)
   self.maskStc = torch.Tensor(outputSize, inputSize)
   self.randmat = torch.Tensor(outputSize, inputSize)
   self.bias = torch.Tensor(outputSize)
   self.gradWeight = torch.Tensor(outputSize, inputSize)
   self.gradBias = torch.Tensor(outputSize)
   self.stcWeights=stcWeights
   self:reset()
end

function BinaryLinear:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.weight:size(2))
   end
   if nn.oldSeed then
      for i=1,self.weight:size(1) do
         self.weight:select(1, i):apply(function()
            return torch.uniform(-1, 1)
         end)
         self.bias[i] = torch.uniform(-stdv, stdv)
      end
   else
      self.weight:uniform(-1, 1)
      self.bias:uniform(-stdv, stdv)
   end

   return self
end

function BinaryLinear:binarized(trainFlag)
  self.weightOrg:copy(self.weight)
  --self.binaryFlag=true
  if not trainFlag and self.stcWeights then
    self.weight:copy(self.weightOrg)
  else
    self.weightB:copy(self.weight):add(1):div(2):clamp(0,1)

    if not self.stcWeights then
      self.weightB:round():mul(2):add(-1)
    else
      self.maskStc=self.weightB-self.randmat:rand(self.randmat:size())
      self.weightB:copy(self.maskStc)

    end
  end

  return  self.weightB
end
function BinaryLinear:updateOutput(input)

  self.weightB = self:binarized(self.train)
   if input:dim() == 1 then
      self.output:resize(self.bias:size(1))
      self.output:copy(self.bias)
      self.output:addmv(1, self.weightB, input)
   elseif input:dim() == 2 then
      local nframe = input:size(1)
      local nElement = self.output:nElement()
      self.output:resize(nframe, self.bias:size(1))
      if self.output:nElement() ~= nElement then
         self.output:zero()
      end
      self.addBuffer = self.addBuffer or input.new()
      if self.addBuffer:nElement() ~= nframe then
         self.addBuffer:resize(nframe):fill(1)
      end
      self.output:addmm(0, self.output, 1, input, self.weightB:t())
      self.output:addr(1, self.addBuffer, self.bias)
   else
      error('input must be vector or matrix')
   end
   self.weight:copy(self.weightOrg);
   return self.output
end

function BinaryLinear:updateGradInput(input, gradOutput)

   if self.gradInput then
      local nElement = self.gradInput:nElement()
      self.gradInput:resizeAs(input)
      if self.gradInput:nElement() ~= nElement then
         self.gradInput:zero()
      end
      if input:dim() == 1 then
         self.gradInput:addmv(0, 1, self.weightB:t(), gradOutput)
      elseif input:dim() == 2 then
         self.gradInput:addmm(0, 1, gradOutput, self.weightB)
      end
      self.gradInput=self.gradInput
      return self.gradInput
   end

end

function BinaryLinear:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   if input:dim() == 1 then
      self.gradWeight:addr(scale, gradOutput, input)
      self.gradBias:add(scale, gradOutput)
   elseif input:dim() == 2 then
      self.gradWeight:addmm(scale, gradOutput:t(), input)
      self.gradBias:addmv(scale, gradOutput:t(), self.addBuffer)
   end
end

-- we do not need to accumulate parameters when sharing
BinaryLinear.sharedAccUpdateGradParameters = BinaryLinear.accUpdateGradParameters


function BinaryLinear:__tostring__()
  return torch.type(self) ..
      string.format('(%d -> %d)', self.weight:size(2), self.weight:size(1))
end
