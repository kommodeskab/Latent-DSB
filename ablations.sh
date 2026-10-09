# python main.py experiment=dsb_beta_ablation model.scheduler.epsilon=0.0 ckpt_filename=290726134720/model-290726134720:v1
# python main.py experiment=dsb_beta_ablation model.scheduler.epsilon=0.1 ckpt_filename=290726194639/model-290726194639:v0
# python main.py experiment=dsb_beta_ablation model.scheduler.epsilon=1.0 ckpt_filename=300726000251/model-300726000251:v0
# python main.py experiment=dsb_beta_ablation model.scheduler.epsilon=2.0 ckpt_filename=300726101423/model-300726101423:v0
# python main.py experiment=dsb_beta_ablation model.scheduler.epsilon=5.0 ckpt_filename=300726143005/model-300726143005:v0

# python main.py experiment=dsb_mamba_blocks_ablation model.model.num_blocks=2  ckpt_filename=300726202115/model-300726202115:v0
# python main.py experiment=dsb_mamba_blocks_ablation model.model.num_blocks=4  ckpt_filename=300726212526/model-300726212526:v0
# python main.py experiment=dsb_mamba_blocks_ablation model.model.num_blocks=6  ckpt_filename=300726231532/model-300726231532:v0
# python main.py experiment=dsb_mamba_blocks_ablation model.model.num_blocks=8  ckpt_filename=310726015654/model-310726015654:v0
# python main.py experiment=dsb_mamba_blocks_ablation model.model.num_blocks=10 ckpt_filename=310726052326/model-310726052326:v0

# python main.py experiment=dsb_inference_steps_ablation model.inference_steps=1  ckpt_filename=310726095039/model-310726095039:v0
# python main.py experiment=dsb_inference_steps_ablation model.inference_steps=2  ckpt_filename=310726110320/model-310726110320:v0
# python main.py experiment=dsb_inference_steps_ablation model.inference_steps=5  ckpt_filename=310726122235/model-310726122235:v0
python main.py experiment=dsb_inference_steps_ablation model.inference_steps=10 ckpt_filename=310726140147/model-310726140147:v0
python main.py experiment=dsb_inference_steps_ablation model.inference_steps=20 ckpt_filename=310726161555/model-310726161555:v0
