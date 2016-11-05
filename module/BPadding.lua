local BPadding, parent = torch.class('nn.BPadding', 'nn.Module')

-- pad puts in [pad] amount of [value] over dimension [dim], starting at index [index] in that dimension. If pad<0, index counts from the left.  If pad>0 index counts from the right
-- index = 1 pads before index 1.  index = 2 pads starting before index 2 and after index 1 in dimension [dim]
function BPadding:__init(dim, pad)
	self.dim = dim
	self.pad = pad
	self.apad = self.pad + 1
	self.outputSize = torch.LongStorage()
	parent.__init(self)
end

function BPadding:updateOutput(input)
	self.outputSize:resize(input:dim())
	self.outputSize:copy(input:size())
	local _idpsize = self.outputSize[self.dim]
	self.outputSize[self.dim] = _idpsize + self.pad * 2
	self.output:resize(self.outputSize)
	self.output:narrow(self.dim, 1, self.pad):zero()
	self.output:narrow(self.dim, _idpsize + self.apad , self.pad):zero()
	self.output:narrow(self.dim, 1 + self.pad, _idpsize):copy(input)
	return self.output
end

function BPadding:updateGradInput(input, gradOutput)
	self.gradInput = gradOutput:narrow(self.dim, self.apad, input:size(self.dim))
	return self.gradInput
end