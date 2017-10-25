local nn = require 'nn'
require 'cunn'

local Convolution = nn.SpatialConvolution
local DilationConv = nn.SpatialDilatedConvolution
local Aug = nn.SpatialAveragePooling
local ReLU = nn.ReLU
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization

local M = {}

-- The shortcut layer is either identity or 1*1 convolution



local depth = 2
local shortcutType = 'B'
local iChannels
local function shortcut(nInputPlane, nOutputPlane, stride, dilation)
	local useConv = shortcutType == 'C' or (shortcutType == 'B' and nInputPlane ~= nOutputPlane) or dilation == 2 or dilation == 4
	if useConv then 
		-- 1*1 convolution
		return nn.Sequential()
			:add(Convolution(nInputPlane, nOutputPlane, 1, 1, stride, stride))
			:add(SBatchNorm(nOutputPlane))
	elseif nInputPlane ~= nOutputPlane then
		-- Strided, zero-padded identity shortcut
		return nn.Sequential()
			:add(nn.SpatialAveragePooling(1, 1, stride, stride))
			:add(nn.Concat(2)
				:add(nn.Identity())
				:add(nn.MulConstant(0)))
	else
		return nn.Identity()
	end
end

local function basicblock(n, stride)
	local nInputPlane = iChannels
	iChannels = n

	local s = nn.Sequential()
	s:add(Convolution(nInputPlane, n, 3, 3, stride, stride, 1, 1))
	s:add(SBatchNorm(n))
	s:add(ReLU(true))
	s:add(Convolution(n, n, 3, 3, 1, 1, 1,1))
	s:add(SBatchNorm(n))

	return nn.Sequential()
		:add(nn.ConcatTable()
			:add(s)
			:add(shortcut(nInputPlane, n, stride, 1)))
		:add(nn.CAddTable(true))
		:add(ReLU(true))
end

local function bottleneck(n, stride, dilation)
	local nInputPlane = iChannels
	iChannels = n * 4
	padding = 1
	if dilation == 2 then
		padding = 2
	elseif dilation == 4 then
		padding = 4
	end

	local s = nn.Sequential()
	s:add(Convolution(nInputPlane, n, 1, 1, stride, stride, 0, 0))
	s:add(SBatchNorm(n))
	s:add(ReLU(true))
	s:add(DilationConv(n, n, 3, 3, 1, 1, padding, padding, dilation, dilation))
	s:add(SBatchNorm(n))
	s:add(ReLU(true))
	s:add(Convolution(n, n*4, 1, 1, 1, 1, 0, 0))
	s:add(SBatchNorm(n*4))

	return nn.Sequential()
		:add(nn.ConcatTable()
			:add(s)
			:add(shortcut(nInputPlane, n*4, stride, dilation)))
		:add(nn.CAddTable(true))
        :add(ReLU(true))
end

local function layer(block, features, count, stride, dilation) --count: the number of block
	local s = nn.Sequential()
	for i=1,count do
		s:add(block(features, i == 1 and stride or 1, dilation))
	end
	return s
end

local function network()
	local model = nn.Sequential()
	iChannels = 64
	layers = {3, 4, 23, 3}
	padding_series = {6, 12, 18, 24}
	dilation_series = {6, 12, 18, 24}
	model:add(Convolution(2, 64, 7, 7, 2, 2, 3, 3))
        --test
        
	model:add(SBatchNorm(64))
	model:add(ReLU(true))
	model:add(Max(3,3,2,2,1,1))
	model:add(layer(bottleneck, 64, layers[1], 1, 1))
        
	model:add(layer(bottleneck, 128, layers[2], 2, 1))
        
	model:add(layer(bottleneck, 256, layers[3], 1, 2))
        
	model:add(layer(bottleneck, 512, layers[4], 1, 4))
                
	model:add(nn.ConcatTable()
		:add(DilationConv(2048, 1, 3, 3, 1, 1, padding_series[1], padding_series[1], dilation_series[1], dilation_series[1]))
		:add(DilationConv(2048, 1, 3, 3, 1, 1, padding_series[2], padding_series[2], dilation_series[2], dilation_series[2]))
		:add(DilationConv(2048, 1, 3, 3, 1, 1, padding_series[3], padding_series[3], dilation_series[3], dilation_series[3]))
		:add(DilationConv(2048, 1, 3, 3, 1, 1, padding_series[4], padding_series[4], dilation_series[4], dilation_series[4])))
        
	model:add(nn.CAddTable(true))
       
    model:add(nn.SpatialUpSamplingBilinear(4))
       
    model:add(nn.Sigmoid())
       
	local function ConvInit(name)
		for k,v in pairs(model:findModules(name)) do
			local n = v.kW * v.kH * v.nOutputPlane
			v.weight:normal(0, math.sqrt(2/n))
			if v.bias then v.bias:zero() end
		end
	end

	local function BNInit(name)
		for k,v in pairs(model:findModules(name)) do
			v.weight:fill(1)
			v.bias:zero()
		end
	end

	--ConvInit('cudnn.SpatialConvolution')
	ConvInit('nn.SpatialConvolution')
	--ConvInit('cudnn.SpatialDilatedConvolution')
	ConvInit('nn.SpatialDilatedConvolution')
	BNInit('fbnn.SpatialBatchNormalization')
	BNInit('cudnn.SpatialBatchNormalization')
	BNInit('nn.SpatialBatchNormalization')


	return model
end


function M.deeplab()
	local input = nn.Identity()()
	local output = network()(input)
	return nn.gModule({input}, {output})
end
return M
