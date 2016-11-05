------------------------------------------------------------------------
--[[ SeqBGRU ]] --
-- Bi-directional RNN using two SeqGRU modules.
-- Input is a tensor e.g time x batch x inputdim.
-- Output is a tensor of the same length e.g time x batch x outputdim.
-- Applies a forward rnn to input tensor in forward order
-- and applies a backward rnn in reverse order.
-- Reversal of the sequence happens on the time dimension.
-- For each step, the outputs of both rnn are merged together using
-- the merge module (defaults to nn.CAddTable() which sums the activations).
------------------------------------------------------------------------
local JSeqBGRU, parent = torch.class('nn.JSeqBGRU', 'nn.Container')

function JSeqBGRU:__init(inputDim, hiddenDim)
    self.forwardModule = nn.SeqGRU(inputDim, hiddenDim)
    self.backwardModule = nn.SeqGRU(inputDim, hiddenDim)
    self.dim = 1
    local backward = nn.Sequential()
    backward:add(nn.SeqReverseSequence(self.dim)) -- reverse
    backward:add(self.backwardModule)
    backward:add(nn.SeqReverseSequence(self.dim)) -- unreverse

    local brnn = nn.Concat(3)
    concat:add(self.forwardModule):add(backward)

    parent.__init(self)

    self.output = torch.Tensor()
    self.gradInput = torch.Tensor()

    self.module = brnn
    -- so that it can be handled like a Container
    self.modules[1] = brnn
end

function JSeqBGRU:updateOutput(input)
    self.output = self.module:updateOutput(input)
    return self.output
end

function JSeqBGRU:updateGradInput(input, gradOutput)
    self.gradInput = self.module:updateGradInput(input, gradOutput)
    return self.gradInput
end

function JSeqBGRU:accGradParameters(input, gradOutput, scale)
    self.module:accGradParameters(input, gradOutput, scale)
end

function JSeqBGRU:accUpdateGradParameters(input, gradOutput, lr)
    self.module:accUpdateGradParameters(input, gradOutput, lr)
end

function JSeqBGRU:sharedAccUpdateGradParameters(input, gradOutput, lr)
    self.module:sharedAccUpdateGradParameters(input, gradOutput, lr)
end

function JSeqBGRU:__tostring__()
    if self.module.__tostring__ then
        return torch.type(self) .. ' @ ' .. self.module:__tostring__()
    else
        return torch.type(self) .. ' @ ' .. torch.type(self.module)
    end
end
