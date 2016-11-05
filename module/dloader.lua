require "cutorch"

function loadObject(fname)
	local file=torch.DiskFile(fname)
	local objRd=file:readObject()
	file:close()
	return objRd
end

--[[function loadseq(fname)
	local file=io.open(fname)
	local num=file:read("*n")
	local rs={}
	while num do
		local tmpt={}
		for i=1,num do
			local vi=file:read("*n")
			table.insert(tmpt,vi)
		end
		table.insert(rs,tmpt)
		num=file:read("*n")
	end
	file:close()
	return rs
end]]

function loadSeqTensor(fname)
	local file=io.open(fname)
	local lind=file:read("*n")
	local rs={}
	local num=file:read("*n")
	while num do
		local tmpt={}
		for i=1,lind do
			table.insert(tmpt,num)
			num=file:read("*n")
		end
		table.insert(rs,torch.LongTensor(tmpt):cuda())
	end
	file:close()
	return rs
end

function loadTrain(iprefix,tprefix,ifafix,tfafix,nfile)
	local id={}
	local td={}
	for i=1,nfile do
		table.insert(id,loadObject(iprefix..i..ifafix):cuda())
		table.insert(td,loadSeqTensor(tprefix..i..tfafix))
	end
	return id,td
end

wvec=loadObject('datasrc/wvec.asc')
sizvec=wvec:size(2)

mword,mwordt=loadTrain('datasrc/thd/train','datasrc/duse/addtag','i.asc','t.txt',243)

devin,devt=loadTrain('datasrc/thd/test','datasrc/duse/msr_test','i.asc','t.txt',110)

nsam=#mword
ndev=#devin

eaddtrain=ieps*nsam
