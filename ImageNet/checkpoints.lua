
local checkpoint = {}

function getModelFlag(name)
  local e=string.len(name)
  local fi,fj=string.find(name, 'imagenet')
  local flag=string.sub(name,fj+2,e)
   return flag
end
function checkpoint.latest(opt)
   if opt.resume == 'none' then
      return nil
   end
   local modelName=getModelFlag(opt.model)
   local Str=modelName..'_depth'..opt.depth..'_b'..opt.batchSize..'_LR'..opt.LR..'_'
   local path_latest=Str..'latest.t7'
   local latestPath = paths.concat(opt.resume, path_latest)
   if not paths.filep(latestPath) then
      return nil
   end

   print('=> Loading checkpoint ' .. latestPath)
   local latest = torch.load(latestPath)
   local optimState = torch.load(latest.optimFile)
   return latest, optimState
end

function checkpoint.save(epoch, model, optimState, bestModel,opt)
   if torch.type(model) == 'nn.DataParallelTable' then
      model = model:get(1)
   end
   local modelName=getModelFlag(opt.model)
   local Str=modelName..'_depth'..opt.depth..'_b'..opt.batchSize..'_LR'..opt.LR..'_'
   local modelFile = 'checkpoints/model_'..Str .. epoch .. '.t7'
   local optimFile = 'checkpoints/optimState_'..Str.. epoch .. '.t7'

   torch.save(modelFile, model)
   torch.save(optimFile, optimState)
   torch.save('checkpoints/'..Str..'latest.t7', {
      epoch = epoch,
      modelFile = modelFile,
      optimFile = optimFile,
   })

   if bestModel then
      torch.save(Str..'model_best.t7', model)
   end
end

return checkpoint
