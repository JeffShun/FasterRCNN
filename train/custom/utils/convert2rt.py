
import onnx
import torch
import argparse
import os, sys
import tensorrt as trt
from onnxsim import simplify
import netron
from onnx import helper
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

work_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(work_dir)
from config.model_config import network_cfg

def load_model(model_path):
    model = network_cfg.network
    checkpoint = torch.load(model_path, map_location={"cuda:0":"cuda:0","cuda:1":"cuda:0","cuda:2":"cuda:0","cuda:3":"cuda:0"})
    model.load_state_dict(checkpoint)
    model = model.cuda()
    model.eval()
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./checkpoints/FasterRCNN-FPN/100.pth')
    parser.add_argument('--output_onnx', type=str, default='./checkpoints/onnx_model')
    parser.add_argument('--output_trt', type=str, default='./checkpoints/trt_model')
    args = parser.parse_args()
    return args


def patch_roi_align(onnx_path):
    model = onnx.load(onnx_path)
    graph = model.graph
    patched = False

    for node in graph.node:
        if node.op_type == "RoiAlign":
            # 查一下是否已有该属性
            names = {a.name for a in node.attribute}
            if "coordinate_transformation_mode" not in names:
                node.attribute.extend([
                    helper.make_attribute(
                        "coordinate_transformation_mode",
                        "align_pytorch"
                    )
                ])
                patched = True

    if patched:
        onnx.save(model, onnx_path)
        print(f"✅ 已补丁并保存到 {onnx_path}")
    else:
        print("ℹ️ 未找到需要补丁的 RoiAlign，输出保持不变")

def torch2onnx(model_path, onnx_path):
    model = load_model(model_path)
    with torch.no_grad():
        # 定义示例输入
        dummy_input = torch.ones((1, 3, 768, 1280),requires_grad=True).cuda()

        # Export the model   
        torch.onnx.export(
            model,                                                 # model being run 
            dummy_input,                                           # model input (or a tuple for multiple inputs) 
            onnx_path,                                             # where to save the model  
            export_params=True,                                    # store the trained parameter weights inside the model file 
            opset_version=17,                                      # the ONNX version to export the model to 
            do_constant_folding=True,                              # whether to execute constant folding for optimization 
            input_names = ['input'],                               # the model's input names 
            output_names = ['output'],                             # the model's output names 
            dynamic_axes = None                                    # dynamic axes
            )
        print('✅ Model has been converted to ONNX!') 

        """
        torchvision.ops.roi_align的属性为coordinate_transformation_mode，默认值为"align_corners"，
        而在低版本onnx中，该属性的值只能为"align_pytorch"，因此需要手动添加该属性。
        """
        patch_roi_align(onnx_path)

        torch.cuda.empty_cache()
        onnx_model = onnx.load(onnx_path)  # load onnx model
        model_simp, check = simplify(onnx_model)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, onnx_path)
        print('✅ Onnx model has been succesfully simplified!')
        # netron.start(onnx_path)

def onnx2trt(onnx_file_path, engine_file_path):
    G_LOGGER = trt.Logger(trt.Logger.WARNING)
    # 1、动态输入第一点必须要写的
    explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    batch_size = 1  # trt推理时最大支持的batchsize
    with trt.Builder(G_LOGGER) as builder, builder.create_network(explicit_batch) as network, trt.OnnxParser(network, G_LOGGER) as parser:
        builder.max_batch_size = batch_size
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 32 
        config.set_flag(trt.BuilderFlag.FP16)
        print('Loading ONNX file from path')
        with open(onnx_file_path, 'rb') as model:
            print('Beginning ONNX file parsing')
            parser.parse(model.read())
        print('Completed parsing of ONNX file')
        print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
        # 重点: 动态输入时候需要 分别为最小输入、常规输入、最大输入
        profile = builder.create_optimization_profile() 
        # 有几个输入就要写几个profile.set_shape 名字和转onnx的时候要对应
        # tensorrt6以后的版本是支持动态输入的，需要给每个动态输入绑定一个profile，用于指定最小值，常规值和最大值，如果超出这个范围会报异常。
        profile.set_shape("input", (1, 3, 768, 1280), (1, 3, 768, 1280), (1, 3, 768, 1280))
        config.add_optimization_profile(profile)
        engine = builder.build_engine(network, config)
        print("Completed creating Engine")
        # 保存engine文件
        with open(engine_file_path, "wb") as f:
            f.write(engine.serialize())

def onnx2trtX(onnx_file_path, engine_file_path):
    os.system("D:/软件包/AI定位软件包/TensorRT-8.4.1.5/bin/trtexec.exe \
              --onnx={} \
              --saveEngine={} \
              --fp16 \
              --workspace=16384 \
              --verbose".format(onnx_file_path, engine_file_path)
              )
    print("Completed creating Engine")

if __name__ == '__main__':
    args = parse_args()
    model_path = args.model_path
    output_onnx_dir = args.output_onnx
    output_trt_dir = args.output_trt
    print("Beigin torch2onnx!")
    os.makedirs(output_onnx_dir, exist_ok=True)
    onnx_path = os.path.join(output_onnx_dir, 'model.onnx')
    torch2onnx(model_path, onnx_path)

    # print("Beigin onnx2trt!")
    # os.makedirs(output_trt_dir, exist_ok=True)
    # engine_file_path = os.path.join(output_trt_dir, 'model.engine')
    # onnx2trtX(onnx_path, engine_file_path)


