require "BPadding"
require "RPadding"
require "getmaxout"
require "getgcnn"
require "rnn"
--require "SeqBLMGRU"
--require "SeqBGRU"
require "SeqDropout"
--require "nngraph"
require "dpnn"
require "vecLookup"
require "maskZerovecLookup"
require "ASequencerCriterion"

function getnn()
	--return getonn()
	return getnnn()
end

function getonn()
	wvec = nil
	local lmod = loadObject("modrs/nnmod.asc").module
	--local lmod = torch.load("modrs/nnmod.asc").module
	return lmod
end

function getnnn()

	local igrusize = sizvec*5;

	local id2vec = nn.maskZerovecLookup(wvec);

	local combm = getgcnn(2,sizvec,3);

	local uniprep = nn.Sequential()
		:add(nn.BPadding(1,2))
		:add(nn.Concat(3)
			:add(nn.Narrow(1,1,-3))
			:add(nn.Narrow(1,2,-2))
			:add(nn.Narrow(1,3,-1)));

	local biprep = nn.Sequential()
		:add(nn.BPadding(1,1))
		:add(nn.Concat(3)
			:add(nn.Narrow(1,1,-2))
			:add(nn.Narrow(1,2,-1)))
		:add(combm);

	local forwardModule = nn.Sequential()
		:add(nn.ParallelTable()
			:add(nn.Narrow(1,3,-1))
			:add(nn.Sequential()
				:add(nn.Narrow(1,2,-1))
				:add(nn.Concat(3)
					:add(nn.Narrow(1,1,-2))
					:add(nn.Narrow(1,2,-1)))
				:add(nn.RPadding(1,1))))
		:add(nn.JoinTable(3,3))
		:add(nn.SeqGRU(igrusize,sizvec));

	forwardModule:get(3).maskzero = true;

	local backwardModule = nn.Sequential()
		:add(nn.ParallelTable()
			:add(nn.Narrow(1,1,-3))
			:add(nn.Sequential()
				:add(nn.Narrow(1,1,-2))
				:add(nn.Concat(3)
					:add(nn.Narrow(1,1,-2))
					:add(nn.Narrow(1,2,-1)))
				:add(nn.RPadding(1,1))))
		:add(nn.JoinTable(3,3))
		:add(nn.SeqReverseSequence(1))
		:add(nn.SeqGRU(igrusize,sizvec))
		:add(nn.SeqReverseSequence(1));

	backwardModule:get(4).maskzero = true;

	local clsmod = nn.Bottle(nn.MaskZero(getmaxout(sizvec*2,nclass,8),1));

	local nms = nn.Sequencer(nn.NormStabilizer());

	local nnmod = nn.Sequential()
		:add(id2vec)
		:add(nn.SeqDropout(0.2))
		:add(nn.ConcatTable()
			:add(uniprep)
			:add(biprep))
		:add(nn.ConcatTable()
			:add(forwardModule)
			:add(backwardModule))
		:add(nn.JoinTable(3,3))
		:add(nms)
		:add(clsmod)
		:add(nn.SplitTable(1));

	return nnmod

end

function getcrit()
	return nn.ASequencerCriterion(nn.MaskZeroCriterion(nn.MultiMarginCriterion(),1));
end

function setupvec(modin,value)
	modin:get(1).updatevec = value
end

function dupvec(modin)
	setupvec(modin,false)
end

function upvec(modin)
	setupvec(modin,true)
end

function setnormvec(modin,value)
	modin:get(1).usenorm = value
end

function dnormvec(modin)
	setnormvec(modin,false)
end

function normvec(modin)
	setnormvec(modin,true)
end