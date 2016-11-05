print("set default tensor type to float")
torch.setdefaulttensortype('torch.FloatTensor')

function gradUpdate(mlpin, x, y, criterionin, learningRate)
	local pred=mlpin:forward(x)
	local err=criterionin:forward(pred, y)
	sumErr=sumErr+err
	local gradCriterion=criterionin:backward(pred, y)
	mlpin:zeroGradParameters()
	mlpin:backward(x, gradCriterion)
	mlpin:updateGradParameters(0.9)
	mlpin:updateParameters(learningRate)
	mlpin:maxParamNorm(2)
end

function evaDev(mlpin, x, y, criterionin)
	mlpin:evaluate()
	local serr=0
	for curpot,v in ipairs(x) do
		serr=serr+criterionin:forward(mlpin:forward(v), y[curpot])
	end
	mlpin:training()
	return serr/ndev
end

--[[function inirand(cyc)
	cyc=cyc or 8
	for i=1,cyc do
		local sdata=math.random(nsam)
	end
end]]

function saveObject(fname,objWrt)
	local tmpod=nil
	if torch.isTensor(objWrt) then
		tmpod=objWrt:float()
	else
		objWrt:float()
		tmpod=nn.Serial(objWrt:clone())
		tmpod:mediumSerial()
		objWrt:cuda()
	end
	local file=torch.DiskFile(fname,'w')
	file:writeObject(tmpod)
	file:close()
end

print("load settings")
require"conf"

print("load data")
require "dloader"

sumErr=0
crithis={}
cridev={}
erate=0
edevrate=0
storemini=1
storedevmini=1
minerrate=starterate
mindeverrate=minerrate

if cycs then
	bnnmod=nil
	bdevnnmod=nil
	bupdnm=false
	bupdevnm=false
end

function train()

	print("prepare environment")
	require "paths"
	local savedir="modrs/"..runid.."/"
	paths.mkdir(savedir)

	print("design neural networks and criterion")

	require "cunn"
	require "cunnx"

	require "designn"
	local nnmod=getnn()

	print(nnmod)
	nnmod:training()

	local critmod=getcrit()

	nnmod:cuda()
	critmod:cuda()

	print("turn off embeddings update")
	dupvec(nnmod)
	print("turn off embeddings norm")
	dnormvec(nnmod)

	print("init train")
	local epochs=1
	local lr=modlr
	--inirand()

	mindeverrate=evaDev(nnmod,devin,devt,critmod)
	print("Init model Dev:"..mindeverrate)

	collectgarbage()

	print("start pre train")
	for tmpi=1,warmcycle do
		for tmpj=1,ieps do
			for curpot,v in ipairs(mword) do
				gradUpdate(nnmod,v,mwordt[curpot],critmod,lr)
			end
		end
		local erate=sumErr/eaddtrain
		if erate<minerrate then
			minerrate=erate
		end
		table.insert(crithis,erate)
		print("epoch:"..tostring(epochs)..",lr:"..lr..",Tra:"..erate)
		sumErr=0
		epochs=epochs+1
	end
	
	print("save neural network trained")
	saveObject(savedir.."nnmod.asc",nnmod)
	
	print("turn on embeddings update")
	upvec(nnmod)

	epochs=1
	icycle=1

	aminerr=1
	lrdecayepochs=1
	
	collectgarbage()

	while true do
		print("start innercycle:"..icycle)
		for innercycle=1,gtraincycle do
			for tmpi=1,ieps do
				for curpot,v in ipairs(mword) do
					gradUpdate(nnmod,v,mwordt[curpot],critmod,lr)
				end
			end
			local erate=sumErr/eaddtrain
			table.insert(crithis,erate)
			local edevrate=evaDev(nnmod,devin,devt,critmod)
			table.insert(cridev,edevrate)
			print("epoch:"..tostring(epochs)..",lr:"..lr..",Tra:"..erate..",Dev:"..edevrate)
			--print("epoch:"..tostring(epochs)..",lr:"..lr..",Tra:"..erate)
			local modsavd=false
			if edevrate<mindeverrate then
				print("new minimal dev error found,save model")
				mindeverrate=edevrate
				if cycs then
					nnmod:float()
					bdevnnmod=nnmod:clone()
					nnmod:cuda()
					bupdevnm=true
				else
					saveObject(savedir.."devnnmod"..storedevmini..".asc",nnmod)
					storedevmini=storedevmini+1
					if storedevmini>csave then
						storedevmini=1
					end
				end
				modsavd=true
			end
			if erate<minerrate then
				minerrate=erate
				aminerr=1
				if not modsavd then
					print("new minimal error found,save model")
					if cycs then
						nnmod:float()
						bnnmod=nnmod:clone()
						nnmod:cuda()
						bupdnm=true
					else
						saveObject(savedir.."nnmod"..storemini..".asc",nnmod)
						storemini=storemini+1
						if storemini>csave then
							storemini=1
						end
					end
				end
			else
				if aminerr>=expdecaycycle then
					aminerr=0
					if lrdecayepochs>lrdecaycycle then
						modlr=lr
						lrdecayepochs=1
					end
					lrdecayepochs=lrdecayepochs+1
					lr=modlr/(lrdecayepochs)
				end
				aminerr=aminerr+1
			end
			sumErr=0
			if cycs and epochs%savecycle==0 then
				if bupdevnm then
					print("flush dev mod")
					saveObject(savedir.."devnnmod"..storedevmini..".asc",bdevnnmod)
					bdevnnmod=nil
					storedevmini=storedevmini+1
					if storedevmini>csave then
						storedevmini=1
					end
					bupdevnm=false
				end
				if bupdnm then
					print("flush mod")
					saveObject(savedir.."nnmod"..storemini..".asc",bnnmod)
					bnnmod=nil
					storemini=storemini+1
					if storemini>csave then
						storemini=1
					end
					bupdnm=false
				end
			end
			epochs=epochs+1
		end

		print("save neural network trained")
		saveObject(savedir.."nnmod.asc",nnmod)

		print("save criterion history trained")
		local critensor=torch.Tensor(crithis)
		saveObject(savedir.."crit.asc",critensor)
		local critdev=torch.Tensor(cridev)
		saveObject(savedir.."critdev.asc",critdev)

		--[[print("plot and save criterion")
		gnuplot.plot(critensor)
		gnuplot.figprint(savedir.."crit.png")
		gnuplot.figprint(savedir.."crit.eps")
		gnuplot.plotflush()
		gnuplot.plot(critdev)
		gnuplot.figprint(savedir.."critdev.png")
		gnuplot.figprint(savedir.."critdev.eps")
		gnuplot.plotflush()]]

		critensor=nil
		critdev=nil

		print("task finished!Minimal error rate:"..minerrate.."	"..mindeverrate)
		--print("task finished!Minimal error rate:"..minerrate)

		print("wait for test, neural network saved at nnmod*.asc")

		icycle=icycle+1

		print("collect garbage")
		collectgarbage()

	end
end

train()
