------------------------------------------------------------------------
--[[ SeqBLMGRU ]] --
-- Bi-directional RNN using two SeqGRU modules.
-- Input is a tensor e.g time x batch x inputdim.
-- Output is a tensor of the same length e.g time x batch x outputdim.
-- Applies a forward rnn to input tensor in forward order
-- and applies a backward rnn in reverse order.
-- Reversal of the sequence happens on the time dimension.
------------------------------------------------------------------------
local SeqBLMGRU, parent = torch.class('nn.SeqBLMGRU', 'nn.Container')

function SeqBLMGRU:__init(inputDim, hiddenDim, maskzero)

	parent.__init(self)

	self.forwardModule = nn.SeqGRU(inputDim, hiddenDim)

	if maskzero then
		self.forwardModule.maskzero=true
	end

	self.backwardModule = nn.Sequential()
		:add(nn.SeqReverseSequence(1)) -- reverse
		:add(self.forwardModule:clone())
		:add(nn.SeqReverseSequence(1)) -- unreverse

	self.modules={self.forwardModule,self.backwardModule}

end

function SeqBLMGRU:updateOutput(input)
	input = input or self.input
	local _pseql = input:size(1)-1
	local _tmptf = self.forwardModule:updateOutput(input:narrow(1,1,_pseql))
	self.output = _tmptf:narrow(1,1,1):clone():zero():cat(_tmptf,1)
	self.output:narrow(1,1,_pseql):add(self.backwardModule:updateOutput(input:narrow(1,2,_pseql)))
	return self.output
end

function SeqBLMGRU:updateGradInput(input, gradOutput)
	input = input or self.input
	gradOutput = gradOutput or self.gradOutput
	local _pseql = gradOutput:size(1)-1
	local _tmptf = self.forwardModule:updateGradInput(input:narrow(1,1,_pseql),gradOutput:narrow(1,2,_pseql))
	self.gradInput = _tmptf:narrow(1,1,1):clone():zero():cat(_tmptf,1)
	self.gradInput:narrow(1,1,_pseql):add(self.backwardModule:updateGradInput(input:narrow(1,2,_pseql),gradOutput:narrow(1,1,_pseql)))
	return self.gradInput
end

function SeqBLMGRU:accGradParameters(input, gradOutput, scale)
	input = input or self.input
	gradOutput = gradOutput or self.gradOutput
	local _pseql = gradOutput:size(1)-1
	self.forwardModule:accGradParameters(input:narrow(1,1,_pseql), gradOutput:narrow(1,2,_pseql), scale)
	self.backwardModule:accGradParameters(input:narrow(1,2,_pseql), gradOutput:narrow(1,1,_pseql), scale)
end

function SeqBLMGRU:accUpdateGradParameters(input, gradOutput, lr)
	input = input or self.input
	gradOutput = gradOutput or self.gradOutput
	local _pseql = gradOutput:size(1)-1
	self.forwardModule:accUpdateGradParameters(input:narrow(1,1,_pseql), gradOutput:narrow(1,2,_pseql), lr)
	self.backwardModule:accUpdateGradParameters(input:narrow(1,2,_pseql), gradOutput:narrow(1,1,_pseql), lr)
end

function SeqBLMGRU:sharedAccUpdateGradParameters(input, gradOutput, lr)
	input = input or self.input
	gradOutput = gradOutput or self.gradOutput
	local _pseql = gradOutput:size(1)-1
	self.forwardModule:sharedAccUpdateGradParameters(input:narrow(1,1,_pseql), gradOutput:narrow(1,2,_pseql), lr)
	self.backwardModule:sharedAccUpdateGradParameters(input:narrow(1,2,_pseql), gradOutput:narrow(1,1,_pseql), lr)
end

function SeqBLMGRU:type(type, ...)
	return parent.type(self, type, ...)
end