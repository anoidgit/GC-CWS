function getmaxout(inputs,outputs,nlinear)
	local ncyc=nlinear or 2
	return nn.Sequential():add(nn.Linear(inputs,outputs*ncyc)):add(nn.Reshape(ncyc,outputs,true)):add(nn.Max(2))
end