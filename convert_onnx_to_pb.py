# from onnx_tf.backend import prepare
# import onnx

# onnx_model = onnx.load("vit_large_patch14_dinov2.onnx")
# tf_rep = prepare(onnx_model)
# tf_rep.export_graph("vit_large_patch14_dinov2.pb")

import onnx2tf

model = onnx2tf.convert("vit_large_patch14_dinov2.onnx")
model.save("vit_large_patch14_dinov2.pb")
