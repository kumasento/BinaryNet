require 'cudnn'
local cudnnBinarySpatialConvolution, parent =
    torch.class('cudnnBinarySpatialConvolution', 'cudnn.SpatialConvolution')
local ffi = require 'ffi'
local errcheck = cudnn.errcheck

local autotunerCache = {}
autotunerCache[1] = {} -- forward
autotunerCache[2] = {} -- backwardFilter
autotunerCache[3] = {} -- backwardData


function cudnnBinarySpatialConvolution:binarized(trainFlagt)
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

function cudnnBinarySpatialConvolution:__init(nInputPlane, nOutputPlane,
                            kW, kH, dW, dH, padW, padH,stcWeights, groups)
    local delayedReset = self.reset
    self.reset = function() end
    parent.__init(self, nInputPlane, nOutputPlane, kW, kH, dW, dH)
    self.reset = delayedReset
    self.padW = padW or 0
    self.padH = padH or 0
    self.stcWeights = stcWeights or false
    self.groups = groups or 1
    assert(nInputPlane % self.groups == 0,
           'nInputPlane should be divisible by nGroups')
    assert(nOutputPlane % self.groups == 0,
           'nOutputPlane should be divisible by nGroups')
    self.weight = torch.Tensor(nOutputPlane, nInputPlane/self.groups, kW, kH)
    self.weightB = torch.Tensor(nOutputPlane, nInputPlane/self.groups, kW, kH)
    self.weightOrg = torch.Tensor(nOutputPlane, nInputPlane/self.groups, kW, kH)
    self.randmat = torch.Tensor(nOutputPlane, nInputPlane/self.groups, kW, kH)
    self.maskStc = torch.Tensor(nOutputPlane, nInputPlane/self.groups, kW, kH)
    self:reset()
    -- should nil for serialization, the reset will still work
    self.reset = nil
    self.iSize = torch.LongStorage(4):fill(0)
end

function cudnnBinarySpatialConvolution:reset(stdv)
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
         return torch.uniform(-stdv, stdv)
      end)
   else
      self.weight:uniform(-1,1)
      self.bias:uniform(-stdv, stdv)
   end
end

-- if you change the configuration of the module manually, call this
function cudnnBinarySpatialConvolution:resetWeightDescriptors()
    assert(torch.typename(self.weight) == 'torch.CudaTensor',
           'Only Cuda supported duh!')
    assert(torch.typename(self.bias) == 'torch.CudaTensor',
           'Only Cuda supported duh!')
    -- for compatibility
    self.groups = self.groups or 1
    -- create filterDescriptor for weight
    self.weightDesc = ffi.new('struct cudnnFilterStruct*[1]')
    errcheck('cudnnCreateFilterDescriptor', self.weightDesc)
    local desc = torch.IntTensor({self.nOutputPlane/self.groups,
                              self.nInputPlane/self.groups,
                              self.kH, self.kW})
    errcheck('cudnnSetFilterNdDescriptor', self.weightDesc[0],
             'CUDNN_DATA_FLOAT', 4,
             desc:data());
    local function destroyWDesc(d)
        errcheck('cudnnDestroyFilterDescriptor', d[0]);
    end
    ffi.gc(self.weightDesc, destroyWDesc)

    -- create descriptor for bias
    self.biasDesc = cudnn.toDescriptor(self.bias:view(1, self.nOutputPlane,1,1))
end


local one = torch.FloatTensor({1});
local zero = torch.FloatTensor({0});

function cudnnBinarySpatialConvolution:updateOutput(input)
    if not self.weightDesc then self:resetWeightDescriptors() end
    self:createIODescriptors(input)

    local prevStream
    local streamQueue = {}
    if self.groups > 1 then -- try to do stream parallelization
        prevStream = cutorch.getStream()
        if prevStream == 0 then
            cutorch.reserveStreams(self.groups)
            for i=1,self.groups do
                cutorch.streamWaitFor(i, {prevStream})
            end
        end
    end

    self.weightOrg:copy(self.weight)
    self.weightB = self:binarized(self.train)
    self.weight:copy(self.weightB)
    for g = 0, self.groups - 1 do
        -- stream-parallelize if appropriate
        if self.groups > 1 and prevStream == 0 then
            cutorch.setStream(g + 1)
            table.insert(streamQueue, g + 1)
        end

        errcheck('cudnnConvolutionForward', cudnn.getHandle(),
                 one:data(),
                 self.iDesc[0], input:data() + g*self.input_offset,
                 self.weightDesc[0], self.weight:data() + g*self.weight_offset,
                 self.convDesc[0], self.fwdAlgType[0],
                 self.extraBuffer:data(), self.extraBufferSizeInBytes,
                 zero:data(),
                 self.oDesc[0], self.output:data() + g*self.output_offset);
    end

    if prevStream == 0 then
        cutorch.setStream(prevStream)
        cutorch.streamWaitFor(prevStream, streamQueue)
    end

    -- add bias
    errcheck('cudnnAddTensor', cudnn.getHandle(),
             'CUDNN_ADD_SAME_C',
             one:data(), self.biasDesc[0], self.bias:data(),
             one:data(), self.oDescForBias[0], self.output:data())
    self.weight:copy(self.weightOrg)
    return self.output
end

function cudnnBinarySpatialConvolution:updateGradInput(input, gradOutput)
  local gradOutput=gradOutput:contiguous()
    if not self.gradInput then return end

    assert(gradOutput:dim() == 3 or gradOutput:dim() == 4, 'gradOutput has to be 3D or 4D');
    assert(gradOutput:isContiguous(), 'gradOutput has to be contiguous')
    if not self.weightDesc then self:resetWeightDescriptors() end
    self:createIODescriptors(input)
    self.weight:copy(self.weightB)

    for g = 0,self.groups - 1 do
        errcheck('cudnnConvolutionBackwardData_v3', cudnn.getHandle(),
                 one:data(),
                 self.weightDesc[0], self.weight:data() + g*self.weight_offset,
                 self.oDesc[0], gradOutput:data() + g*self.output_offset,
                 self.convDesc[0],
                 self.bwdDataAlgType[0],
                 self.extraBuffer:data(), self.extraBufferSizeInBytes,
                 zero:data(),
                 self.iDesc[0], self.gradInput:data() + g*self.input_offset);
    end
    self.weight:copy(self.weightOrg)
    return self.gradInput
end
function cudnnBinarySpatialConvolution:accGradParameters(input, gradOutput, scale)
    parent.accGradParameters(self, input, gradOutput:contiguous(), scale)
end
