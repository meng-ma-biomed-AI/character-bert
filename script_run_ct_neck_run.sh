#!bin/sh
conda activate env1
python3 /home/tpsanto/Github/character-bert/main.py --task classification --data_type ct --data_subtype neck --embedding medical_character_bert --do_lower_case --do_train --do_predict --train_batch_size=8 --eval_batch_size 16 --num_train_epochs 20 --gradient_accumulation_steps 4'
