
python3 examples/data_preprocess/math_dataset_refactored.py --local_dir ~/data/math

python3 -c "import transformers; transformers.pipeline('text-generation', model='Qwen/Qwen2.5-7B-Instruct')"

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
	 data.train_files=$HOME/data/math/train.parquet \
	  data.val_files=$HOME/data/math/test.parquet \
	   data.train_batch_size=256 \
	    data.max_prompt_length=4000 \
	     data.max_response_length=8000 \
	      actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
	       actor_rollout_ref.actor.optim.lr=1e-6 \
	        actor_rollout_ref.actor.ppo_mini_batch_size=64 \
		 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
		  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
		   actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
		    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
		     actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
		      critic.optim.lr=1e-5 \
		       critic.model.path=Qwen/Qwen2.5-7B-Instruct \
		        critic.ppo_micro_batch_size_per_gpu=1 \
			 algorithm.kl_ctrl.kl_coef=0.001 \
			  trainer.logger=['console','wandb'] \
			   +trainer.val_before_train=False \
			     trainer.project_name='project_ppo_qwen2.5-7b' \
    		      trainer.experiment_name='experiment_ppo_qwen2.5-7b-it' \
			       trainer.default_hdfs_dir=null \
			        trainer.n_gpus_per_node=4 \
			         trainer.nnodes=1 \
			          trainer.save_freq=10 \
			           trainer.test_freq=10 \
				        trainer.total_epochs=15 2>&1 | tee verl_demo_qwen.log

