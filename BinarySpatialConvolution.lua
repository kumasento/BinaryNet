local BinarySpatialConvolution, parent = torch.class('BinarySpatialConvolution', 'nn.Module')

function BinarySpatialConvolution:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW,padH)
   parent.__init(self)

   dW = dW or 1
   dH = dH or 1

   self.nInputPlane = nInputPlane
   self.nOutputPlane = nOutputPlane
   self.kW = kW
   self.kH = kH

   self.dW = dW
   self.dH = dH
   self.padding = padding or 0
   self.padW = padW or 0
   self.padH = padH or 0

   self.weight = torch.Tensor(nOutputPlane, nInputPlane, kH, kW)
   self.weightB = torch.Tensor(nOutputPlane, nInputPlane, kH, kW)
   self.weightOrg = torch.Tensor(nOutputPlane, nInputPlane, kH, kW)
   self.randmat = torch.Tensor(nOutputPlane, nInputPlane, kH, kW)
   self.maskStc = torch.Tensor(nOutputPlane, nInputPlane, kH, kW)
   self.bias = torch.Tensor(nOutputPlane)
   self.gradWeight = torch.Tensor(nOutputPlane, nInputPlane, kH, kW)
   self.gradBias = torch.Tensor(nOutputPlane)

   self:reset()
end

function BinarySpatialConvolution:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1/math.sqrt(self.kW*self.kH*self.nInputPlane)
   end
   if nn.oldSeed then
      self.weight:apply(function()
         return torch.uniform(-1, 1)
      end)
      self.bias:apply(function()
         return torch.uniform(-1, 1)
      end)
   else
      self.weight:uniform(-1,1)
      self.bias:uniform(-1, 1)
   end
end

local function binarized(trainFlag)
  self.weightOrg:copy(self.weight);
  if not trainFlag and  self.stcWeights then
    self.weightB:copy(weightOrg);
  else
    self.weightB:copy(self.weight):add(1):div(2):clamp(0,1)

    if not stcWeights then
      self.weightB:round():mul(2):add(-1)
    else
      self.maskStc=self.weightB-self.randmat:rand(self.randmat:size())
      self.weightB:copy(self.maskStc:sign())
    end
  end

  return  self.weightB
  end

  return  weightB
end

local function backCompatibility(self)
   self.finput = self.finput or self.weight.new()
   self.fgradInput = self.fgradInput or self.weight.new()
   self.padding = self.padding or 0
   if self.weight:dim() == 2 then
      self.weight = self.weight:view(self.nOutputPlane, self.nInputPlane, self.kH, self.kW)
   end
   if self.gradWeight and self.gradWeight:dim() == 2 then
      self.gradWeight = self.gradWeight:view(self.nOutputPlane, self.nInputPlane, self.kH, self.kW)
   end
end

local function makeContiguous(self, input, gradOutput)
   if not input:isContiguous() then
      self._input = self._input or input.new()
      self._input:resizeAs(input):copy(input)
      input = self._input
   end
   if gradOutput then
      if not gradOutput:isContiguous() then
	 self._gradOutput = self._gradOutput or gradOutput.new()
	 self._gradOutput:resizeAs(gradOutput):copy(gradOutput)
	 gradOutput = self._gradOutput
      end
   end
   return input, gradOutput
end

-- function to re-view the weight layout in a way that would make the MM ops happy
local function viewWeight(self)
   self.weight = self.weight:view(self.nOutputPlane, self.nInputPlane * self.kH * self.kW)
   if self.gradWeight and self.gradWeight:dim() > 0 then
      self.gradWeight = self.gradWeight:view(self.nOutputPlane, self.nInputPlane * self.kH * self.kW)
   end
end

local function unviewWeight(self)
   self.weight = self.weight:view(self.nOutputPlane, self.nInputPlane, self.kH, self.kW)
   if self.gradWeight and self.gradWeight:dim() > 0 then
      self.gradWeight = self.gradWeight:view(self.nOutputPlane, self.nInputPlane, self.kH, self.kW)
   end
end

function BinarySpatialConvolution:updateOutput(input)
  --print('In u')
   backCompatibility(self)
   viewWeight(self)
   input = makeContiguous(self, input)
   self.weightB = binarized(self.train)
   self.weight:copy(self.weightB)
   local out = input.nn.SpatialConvolutionMM_updateOutput(self, input)
   self.weight:copy(self.weightOrg)
   unviewWeight(self)
   return out
end

function BinarySpatialConvolution:updateGradInput(input, gradOutput)
   if self.gradInput then
      backCompatibility(self)
      viewWeight(self)
      input, gradOutput = makeContiguous(self, input, gradOutput)

      self.weight:copy(self.weightB)
      local out = input.nn.SpatialConvolutionMM_updateGradInput(self, input, gradOutput)
      self.weight:copy(self.weightOrg)

      unviewWeight(self)
      return out
   end
end

function BinarySpatialConvolution:accGradParameters(input, gradOutput, scale)
   backCompatibility(self)
   input, gradOutput = makeContiguous(self, input, gradOutput)
   viewWeight(self)
   local out = input.nn.SpatialConvolutionMM_accGradParameters(self, input, gradOutput, scale)
   unviewWeight(self)
   return out
end

function BinarySpatialConvolution:type(type)
   self.finput = torch.Tensor()
   self.fgradInput = torch.Tensor()
   return parent.type(self,type)
end
