local Linear_AllMode, parent = torch.class('nn.Linear_AllMode', 'nn.Module')

function Linear_AllMode:__init(inputSize, outputSize,InitFlag,flag_updateWeight, interval)
   parent.__init(self)
  
   if flag_updateWeight ~= nil then
      self.flag_updateWeight = flag_updateWeight
   else
      self.flag_updateWeight = true
   end


   self.weight = torch.Tensor(outputSize, inputSize)
   self.bias = torch.Tensor(outputSize)
   self.gradBias = torch.Tensor(outputSize)
   self.gradWeight = torch.Tensor(outputSize, inputSize)
   
    self.correlate=torch.Tensor(inputSize,inputSize)
    self.conditionNumber_input={}
    self.maxEig_input={}
    self.eig_input={}
    self.eig_weight={}
    self.norm_weight={}
    self.epcilo=10^-100
   
    self.correlate_gradOutput=torch.Tensor(outputSize,outputSize)
    self.conditionNumber_gradOutput={}
    self.maxEig_gradOutput={}
    self.eig_gradOutput={}

-- print(InitFlag)
  if InitFlag ~= nil then
      self.InitFlag=InitFlag    
  else
      self.InitFlag='RandInit'    
  end

  print(flag_updateWeight)
  print('InitMehtod:'..self.InitFlag)
  if self.InitFlag=='OrthInit' then
      self:reset_OrthInit() 
   elseif self.InitFlag=='HeInit' then
      self:reset_HeInit() 
   else
       self:reset()
   end
   
   self.debug=true
   self.count=0
   self.interval=interval or 20

end

function Linear_AllMode:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.weight:size(2))
   end
   if nn.oldSeed then
      for i=1,self.weight:size(1) do
         self.weight:select(1, i):apply(function()
            return torch.uniform(-stdv, stdv)
         end)
        if self.isBias then 
         self.bias[i] = torch.uniform(-stdv, stdv)
        end
     end
   else
      self.weight:uniform(-stdv, stdv)
       self.bias:uniform(-stdv, stdv)
    end
   return self
end

function Linear_AllMode:reset_HeInit()
  print('-----------HeInit-------------------')
  local stdv = math.sqrt(2/self.weight:size(2))
  self.weight:normal(0, stdv)
  self.bias:zero()
end

function Linear_AllMode:reset_OrthInit()
  print('-----------OrthInit-------------------')
    local initScale = 1 -- math.sqrt(2)
   -- local initScale =  math.sqrt(2)
    local M1 = torch.randn(self.weight:size(1), self.weight:size(1))
    local M2 = torch.randn(self.weight:size(2), self.weight:size(2))

    local n_min = math.min(self.weight:size(1), self.weight:size(2))
    -- QR decomposition of random matrices ~ N(0, 1)
    local Q1, R1 = torch.qr(M1)
    local Q2, R2 = torch.qr(M2)
    self.weight:copy(Q1:narrow(2,1,n_min) * Q2:narrow(1,1,n_min)):mul(initScale)
    self.bias:zero()
end

function Linear_AllMode:updateOutput(input)
   --self.bias:fill(0)
   ------------------------calcluate the eig and FIM for Input---
    if self.train and self.debug and (self.count % self.interval)==0 then
       -----------------------------for the input--------------
       self.correlate:addmm(0,self.correlate, 1/input:size(1),input:t(), input)
       _,self.buffer,_=torch.svd(self.correlate) 
      table.insert(self.eig_input,self.buffer:clone())
       self.buffer:add(self.epcilo)
       local maxEig=torch.max(self.buffer)
       local conditionNumber=torch.abs(maxEig/torch.min(self.buffer))
       local normWeight=self.weight:norm()
       print('Linear module: input conditionNumber='..conditionNumber..'---maxEig:'..maxEig..'--NormWeight:'..normWeight)
       self.conditionNumber_input[#self.conditionNumber_input + 1]=conditionNumber
       self.maxEig_input[#self.maxEig_input + 1]=maxEig
       self.norm_weight[#self.norm_weight + 1]=normWeight
    end 

   if input:dim() == 1 then
      self.output:resize(self.bias:size(1))
      self.output:copy(self.bias)
      self.output:addmv(1, self.weight, input)
   elseif input:dim() == 2 then
      local nframe = input:size(1)
      local nElement = self.output:nElement()
      self.output:resize(nframe, self.weight:size(1))
      if self.output:nElement() ~= nElement then
         self.output:zero()
      end
      self.addBuffer = self.addBuffer or input.new()
      if self.addBuffer:nElement() ~= nframe then
         self.addBuffer:resize(nframe):fill(1)
      end
      self.output:addmm(0, self.output, 1, input, self.weight:t())
     

      if self.isBias then
        self.output:addr(1, self.addBuffer, self.bias)
      end

   else
      error('input must be vector or matrix')
   end
      
    if self.train and self.debug and (self.count % self.interval)==0 then
     -- local validate_Matrix=self.W* self.W:t()     
     local _,buffer,_=torch.svd(self.weight) --SVD Decompositon for singular value
      --print(buffer)
      table.insert(self.eig_weight,buffer:clone())
    end

   collectgarbage()

   return self.output
end

function Linear_AllMode:updateGradInput(input, gradOutput)
    if self.train and self.debug and (self.count % self.interval)==0 then
       -----------------------------for the gradOutput--------------
       self.correlate_gradOutput:addmm(0,self.correlate_gradOutput, gradOutput:size(1),gradOutput:t(), gradOutput)  --note, for gradOutput, owe 1/N are used in gradOutput, therefore, here 1/N dx dx = 1/N* N^2 d'x d'x = N *d'x d'x, is different to x*x.
       _,self.buffer,_=torch.svd(self.correlate_gradOutput) 
      table.insert(self.eig_gradOutput,self.buffer:clone())
       self.buffer:add(self.epcilo)
       local maxEig=torch.max(self.buffer)
       local conditionNumber=torch.abs(maxEig/torch.min(self.buffer))
       print('Linear module: grdOutput conditionNumber='..conditionNumber..'---maxEig:'..maxEig)
       self.conditionNumber_gradOutput[#self.conditionNumber_gradOutput + 1]=conditionNumber
       self.maxEig_gradOutput[#self.maxEig_gradOutput + 1]=maxEig
    end 
  
   self.count=self.count+1

   if self.gradInput then
      local nElement = self.gradInput:nElement()
      self.gradInput:resizeAs(input)
      if self.gradInput:nElement() ~= nElement then
         self.gradInput:zero()
      end
      if input:dim() == 1 then
         self.gradInput:addmv(0, 1, self.weight:t(), gradOutput)
      elseif input:dim() == 2 then
         self.gradInput:addmm(0, 1, gradOutput, self.weight)
      end
      
      return self.gradInput
   end
end

function Linear_AllMode:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   if input:dim() == 1 then
      self.gradWeight:addr(scale, gradOutput, input)
      self.gradBias:add(scale, gradOutput)
   elseif input:dim() == 2 then
      self.gradWeight:addmm(scale, gradOutput:t(), input)
     
      if self.isBias then
        self.gradBias:addmv(scale, gradOutput:t(), self.addBuffer)
      end
   end

end

function Linear_AllMode:parameters()
    if self.flag_updateWeight then
      return {self.weight, self.bias}, {self.gradWeight, self.gradBias}
    else
      return {}, {}
    end
end

-- we do not need to accumulate parameters when sharing
Linear_AllMode.sharedAccUpdateGradParameters = Linear_AllMode.accUpdateGradParameters



function Linear_AllMode:__tostring__()
  return torch.type(self) ..
      string.format('(%d -> %d)', self.weight:size(2), self.weight:size(1))
end
