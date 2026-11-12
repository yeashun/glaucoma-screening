import torch

def box_from_map(fmap,weight,bbox,input_size,fraction=.45):
    x0,y0,x1,y1=[int(v) for v in bbox]
    width,height=x1-x0,y1-y0
    scale=min(input_size/width,input_size/height)
    offset_x=(input_size-width*scale)/2
    offset_y=(input_size-height*scale)/2
    heat=torch.einsum('c,chw->hw',weight.reshape(-1).to(fmap),fmap.float())
    heat=torch.nn.functional.interpolate(heat[None,None],size=(input_size,input_size),mode='bilinear',align_corners=False)[0,0]
    heat[:max(0,int(offset_y)),:]=-torch.inf
    heat[min(input_size,int(input_size-offset_y)):,:]=-torch.inf
    heat[:,:max(0,int(offset_x))]=-torch.inf
    heat[:,min(input_size,int(input_size-offset_x)):]=-torch.inf
    if not torch.isfinite(heat).any() or heat.max()<=0:
        cx,cy=(x0+x1)/2,(y0+y1)/2
    else:
        index=int(heat.argmax())
        cy=y0+(index//input_size-offset_y)/scale
        cx=x0+(index%input_size-offset_x)/scale
    side=max(8,int(min(width,height)*fraction))
    left=max(x0,min(int(cx-side/2),x1-side))
    top=max(y0,min(int(cy-side/2),y1-side))
    return left,top,left+side,top+side
