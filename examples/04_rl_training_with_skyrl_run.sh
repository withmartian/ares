# My key
export DAYTONA_API_KEY=YOUR_KEY_HERE
export WANDB_API_KEY=YOUR_KEY_HERE
export FLASHINFER_DISABLE_VERSION_CHECK=1

CKPTS_DIR=YOUR_CKPTS_DIR_HERE
EXPORTS_DIR=YOUR_EXPORTS_DIR_HERE

# Run SkyRL command
uv run --isolated --extra vllm --extra harbor -m examples.04_rl_training_with_skyrl \
  +data.ares_preset_name_train=sbv-mswea \
  +data.ares_preset_name_val=tbench-mswea \
  trainer.policy.model.path=Qwen/Qwen3-4B-Instruct-2507 \
  generator.served_model_name=Qwen3-4B-Instruct-2507 \
  trainer.export_path=$EXPORTS_DIR \
  trainer.ckpt_path=$CKPTS_DIR \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_nodes=1 \
  trainer.placement.ref_num_nodes=1 \
  trainer.placement.policy_num_gpus_per_node=8 \
  trainer.placement.ref_num_gpus_per_node=8 \
  generator.num_inference_engines=8 \
  generator.inference_engine_tensor_parallel_size=1 \
  trainer.epochs=3 \
  trainer.eval_batch_size=128 \
  trainer.eval_before_train=true \
  trainer.eval_interval=20 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=64 \
  trainer.policy_mini_batch_size=64 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval=5 \
  trainer.hf_save_interval=5 \
  trainer.max_prompt_length=2048 \
  generator.sampling_params.max_generate_length=30720 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  generator.n_samples_per_prompt=8 \
  generator.eval_n_samples_per_prompt=8 \
  generator.gpu_memory_utilization=0.8 \
  trainer.logger=wandb \
  trainer.project_name=dc-agent \
  trainer.run_name=otagent-rl \
  trainer.resume_mode=latest \
  generator.backend=vllm \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=false \
  generator.enable_http_endpoint=true \
  generator.http_endpoint_host=127.0.0.1 \
  generator.http_endpoint_port=8000
