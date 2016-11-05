local RPadding, parent = torch.class('nn.RPadding', 'nn.Module')

-- pad puts in [pad] amount of [value] over dimension [dim], starting at index [index] in that dimension. If pad<0, index counts from the left.  If pad>0 index counts from the right
-- index = 1 pads before index 1.  index = 2 pads starting before index 2 and after index 1 in dimension [dim]
function RPadding:__init(dim, pad)
	self.dim = dim
	self.pad = pad
	self.outputSize = torch.LongStorage()
	parent.__init(self)
end

function RPadding:updateOutput(input)
	self.outputSize:resize(input:dim())
	self.outputSize:copy(input:size())
	local _idpsize = self.outputSize[self.dim]
	self.outputSize[self.dim] = _idpsize + self.pad
	self.output:resize(self.outputSize)
	self.output:narrow(self.dim, _idpsize + 1, self.pad):zero()
	self.output:narrow(self.dim, 1, _idpsize):copy(input)
	return self.output
end

function RPadding:updateGradInput(input, gradOutput)
	self.gradInput = gradOutput:narrow(self.dim, 1, input:size(self.dim))
	return self.gradInput
end