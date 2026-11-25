import numpy as np
from common import *

def shapes(seed,n):
    r=rng(seed);images=r.normal(0,.08,(n,1,16,16)).astype('float32');masks=np.zeros_like(images)
    for i in range(n):
        a,b=r.integers(2,9,2);h,w=r.integers(3,7,2);masks[i,0,a:a+h,b:b+w]=1;images[i]+=masks[i]*.8
    return images,masks
def run():
    import torch,onnx,onnxruntime as ort
    from onnx import helper,TensorProto,numpy_helper
    torch.manual_seed(16);torch.set_num_threads(2);x,y=shapes(2,100);tx,ty=shapes(99,30)
    model=torch.nn.Conv2d(1,1,3,padding=1);opt=torch.optim.Adam(model.parameters(),lr=.08)
    a=torch.tensor(x);b=torch.tensor(y)
    for _ in range(100):
        logits=model(a);loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,b)
        opt.zero_grad();loss.backward();opt.step()
    with torch.no_grad():pred=model(torch.tensor(tx)).numpy()
    mask=pred>=0;truth=ty>0;iou=float(np.sum(mask&truth)/max(1,np.sum(mask|truth)))
    weight=model.weight.detach().numpy();bias=model.bias.detach().numpy()
    graph=helper.make_graph([helper.make_node('Conv',['x','weight','bias'],['y'],pads=[1,1,1,1])],'tiny-seg',
        [helper.make_tensor_value_info('x',TensorProto.FLOAT,['batch',1,16,16])],
        [helper.make_tensor_value_info('y',TensorProto.FLOAT,['batch',1,16,16])],
        [numpy_helper.from_array(weight,'weight'),numpy_helper.from_array(bias,'bias')])
    export=helper.make_model(graph,opset_imports=[helper.make_opsetid('',17)]);export.ir_version=10;onnx.checker.check_model(export)
    session=ort.InferenceSession(export.SerializeToString(),providers=['CPUExecutionProvider']);actual=session.run(None,{'x':tx})[0]
    return {'test_pixel_iou':iou,'onnx_max_logit_difference':float(abs(actual-pred).max()),'onnx_bytes':len(export.SerializeToString()),
            'checks':[check(np.allclose(actual,pred,atol=1e-5),'Actual ONNX Runtime logits match trained PyTorch convolution'),check(len(tx)==30,'Separate seed for held-out shapes')],
            'scope':'One-layer trained segmentation on authored procedural rectangles. No real-world defect or medical image accuracy claim.'}
