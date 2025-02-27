
python3 examples/data_preprocess/math_dataset_refactored.py --local_dir ~/data/math 

python3 -c "import transformers; transformers.pipeline('text-generation', model='meta-llama/Llama-3.1-8B')" 

set -x export VLLM_ATTENTION_BACKEND=XFORMERS 

python3 -m verl.trainer.main_ppo \
   	algorithm.adv_estimator=grpo \ 
	data.train_files=$HOME/data/math/train.parquet \ 
	data.val_files=$HOME/data/math/test.parquet \ 
	data.train_batch_size=256 \ 
	data.max_prompt_length=8000 \ 
    	data.max_response_length=16000 \ 
	actor_rollout_ref.model.path=meta-llama/Llama-3.1-8B \ 
	actor_rollout_ref.actor.optim.lr=1e-6 \ 
	actor_rollout_ref.model.use_remove_padding=True \ 
    	actor_rollout_ref.actor.ppo_mini_batch_size=256 \ 	
	actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=80 \ 
	actor_rollout_ref.actor.use_kl_loss=True \ 
    	actor_rollout_ref.actor.kl_loss_coef=0.001 \ 
	actor_rollout_ref.actor.kl_loss_type=low_var_kl \ 
	actor_rollout_ref.model.enable_gradient_checkpointing=True \
      	actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \ 
	actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=160 \ 
    	actor_rollout_ref.rollout.tensor_model_parallel_size=2 \ 
	actor_rollout_ref.rollout.name=vllm \ 
	actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \ 
	actor_rollout_ref.rollout.n=5 \ 
    	actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=160 \ 
	actor_rollout_ref.ref.fsdp_config.param_offload=True \ 
	algorithm.kl_ctrl.kl_coef=0.001 \ 
	trainer.critic_warmup=0 \ 
    	trainer.logger=['console','wandb'] \ 
	trainer.project_name='project_grpo_llama'\ 
	trainer.experiment_name='experiment_grpo_llama' \ 
	trainer.n_gpus_per_node=8 \ 
	trainer.nnodes=1 \ 
	trainer.save_freq=5 \ 
	trainer.test_freq=5\
    	trainer.total_epochs=15 2>&1 | tee verl_llama_grpo_demo.log
