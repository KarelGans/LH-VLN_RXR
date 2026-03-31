from utils.agent import HabitatAgent, save_checkpoint
from utils.SFT_agent import SFTAgent
from utils.metrics import NavigationMetrics
from utils.dataset import TaskDataset, EpisodeDataset, create_split_datasets
from utils.parser import read_args, random_seed
from torch.utils.data import DataLoader
from NavModel.RandomNav import RandomAgent
from NavModel.LLMModel.continuous_nav import ContinuousNav
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import numpy as np
import tqdm
import time
import torch


def train_one_sft_epoch(
        args, 
        epoch, 
        dataloader, 
        optimizer,
        lr_scheduler,
        criterion,
        nav_model,
        logger
        ):
    nav_model.model.train()
    lh_losses = []
    st_losses = []

    num_batches_per_epoch = len(dataloader)
    total_training_steps = num_batches_per_epoch * args.num_epochs
    pbar = tqdm.tqdm(
        range(num_batches_per_epoch),
        disable=True,
        total=total_training_steps,
        initial=(epoch * num_batches_per_epoch)
    )
    if args.tensorboard:
        writer_step = SummaryWriter(log_dir=args.tensorboard_path, filename_suffix='epoch_'+str(epoch))
    else:
        writer_step = None
    
    for step, (config, step_configs) in enumerate(dataloader):
        logger.info(f"****** statrt training in step: {step} ******")
        logger.info(config)

        # SFT for LH task
        agent = SFTAgent(args, config, nav_model)
        lh_loss = agent.train(criterion)

        # #training for step task
        st_loss = []
        for step_config in step_configs:
            logger.info(step_config)
            agent = SFTAgent(args, step_config, nav_model)
            st = agent.train(criterion, step_task=True)
            st_loss.append(st)

        torch.nn.utils.clip_grad_norm_(nav_model.model.parameters(), 40.)
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()
        
        verbose_dict = dict(
            step=step,
            loss=lh_loss,
            lr=lr_scheduler.get_last_lr()[0],
        )
        pbar.set_postfix(verbose_dict)
        pbar.update()

        lh_losses.append(lh_loss)
        st_losses.append(sum(st_loss)/len(st_loss) if len(st_loss) > 0 else [])

        if writer_step:
            writer_step.add_scalars('Training loss', {
                'imiation loss': lh_loss,
                'st_loss': sum(st_loss)/len(st_loss) if len(st_loss) > 0 else float("inf"),
            }, step)

        if step % args.save_ckpt_per_epochs == 0 and step != 0:
            checkpoint_path = args.save_checkpoints + '/' + 'epoch_' + str(epoch) + 'step_' + str(step) + '.pth'
            save_checkpoint(nav_model.model, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")

    if writer_step:
        writer_step.close()

    return sum(lh_losses)/len(lh_losses) if len(lh_losses) > 0 else float("inf"), sum(st_losses)/len(st_losses) if len(st_losses) > 0 else float("inf")

def train_one_epoch(
        args, 
        epoch, 
        metrics, 
        dataloader, 
        optimizer,
        lr_scheduler,
        criterion,
        nav_model,
        logger,
        resume_from_step
        ):
    
    nav_model.model.train()
    lh_losses = []
    st_losses = []

    num_batches_per_epoch = len(dataloader)
    total_training_steps = num_batches_per_epoch * args.num_epochs
    pbar = tqdm.tqdm(
        range(num_batches_per_epoch),
        disable=True,
        total=total_training_steps,
        initial=(epoch * num_batches_per_epoch)
    )
    if args.tensorboard:
        writer_step = SummaryWriter(log_dir=args.tensorboard_path, filename_suffix='epoch_'+str(epoch))
    else:
        writer_step = None
    
    for step, (config, step_configs) in enumerate(dataloader):

        if step < args.resume_from_step:
            print(f'SKIPPING STEP................................................................................................................. {step}')
            pbar.update() # <--- This "pushes" the bar forward without doing the work
            continue

        logger.info(f"****** statrt training in step: {step} ******")
        logger.info(config)

        # imiation learning for LH task
        agent = HabitatAgent(args, config, nav_model)
        lh_loss, result = agent.train(criterion)
        if lh_loss == None and result == None:
            logger.warning(f"Skipping step {step} due to simulator/pathfinding error.")
        pbar.update() 
        continue


        metrics[str(len(result['successes']))].add_sample(
            all(result['successes']),
            sum(result['gt_step']),
            sum(result['navigation_steps']),
            all(result['oracle_successes']),
            sum(result['navigation_errors'])/len(result['navigation_errors']) if len(result['navigation_errors']) > 0 else 0,
            result['successes'],
            result['gt_step'],
            result['gt_path'],
            result['navigation_errors']
            )
        metrics[config["Robot"]].add_sample(
            all(result['successes']),
            sum(result['gt_step']),
            sum(result['navigation_steps']),
            all(result['oracle_successes']),
            sum(result['navigation_errors'])/len(result['navigation_errors']) if len(result['navigation_errors']) > 0 else 0,
            result['successes'],
            result['gt_step'],
            result['gt_path'],
            result['navigation_errors']
            )
        metrics['result'].add_sample(
            all(result['successes']),
            sum(result['gt_step']),
            sum(result['navigation_steps']),
            all(result['oracle_successes']),
            sum(result['navigation_errors'])/len(result['navigation_errors']) if len(result['navigation_errors']) > 0 else 0,
            result['successes'],
            result['gt_step'],
            result['gt_path'],
            result['navigation_errors']
            )
        
        # added logging
        current_step_metrics = metrics['result'].compute()
        logger.info(f"[Step {step}] SR: {current_step_metrics['success_rate']:.4f} | "
                    f"SPL: {current_step_metrics['spl']:.4f} | "
                    f"NE: {current_step_metrics['navigation_error']:.4f} | "
                    f"ISR: {current_step_metrics['independent_success_rate']:.4f} | "
                    f"CSR: {current_step_metrics['conditional_success_rate']:.4f}")

        # #training for step task
        st_loss = []
        for step_config in step_configs:
            logger.info(step_config)
            agent = HabitatAgent(args, step_config, nav_model)
            st, res = agent.train(criterion, step_task=True)
            st_loss.append(st)
        
        if (step + 1) % args.gradient_accumulation_step == 0:
            torch.nn.utils.clip_grad_norm_(nav_model.model.parameters(), 40.)
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
        
        verbose_dict = dict(
            step=step,
            loss=lh_loss,
            lr=lr_scheduler.get_last_lr()[0],
        )
        pbar.set_postfix(verbose_dict)
        pbar.update()

        lh_losses.append(lh_loss)
        # change [] tpo 0.0 because of no st
        # st_losses.append(sum(st_loss)/len(st_loss) if len(st_loss) > 0 else [])
        st_losses.append(sum(st_loss)/len(st_loss) if len(st_loss) > 0 else 0.0)


        if writer_step:
            writer_step.add_scalars('Training loss', {
                'imiation loss': lh_loss,
                'st_loss': sum(st_loss)/len(st_loss) if len(st_loss) > 0 else float("inf"),
            }, step)

        if step % args.save_ckpt_per_epochs == 0 and step != 0:
            checkpoint_path = args.save_checkpoints + '/' + 'epoch_' + str(epoch) + 'step_' + str(step) + '.pth'
            save_checkpoint(nav_model.model, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")

    if writer_step:
        writer_step.close()

    return sum(lh_losses)/len(lh_losses) if len(lh_losses) > 0 else float("inf"), sum(st_losses)/len(st_losses) if len(st_losses) > 0 else float("inf")


def validate_one_epoch(
        args, 
        epoch, 
        metrics, 
        dataloader, 
        nav_model,
        logger,
        ):
    
    nav_model.model.eval()

    num_batches_per_epoch = len(dataloader)
    total_training_steps = num_batches_per_epoch * args.num_epochs
    pbar = tqdm.tqdm(
        range(num_batches_per_epoch),
        disable=True,
        total=total_training_steps,
        initial=(epoch * num_batches_per_epoch)
    )
    # test for LH-VLN task and step task 
    for step, (config, step_configs) in enumerate(dataloader):
        logger.info(f"--- STARTING LH TASK: {config['Task instruction'][:70]}... ---")
        logger.info(config)
        # test for LH-task
        agent = HabitatAgent(args, config, nav_model)
        result = agent.validate()
        metrics[str(len(result['successes']))].add_sample(
            all(result['successes']),
            sum(result['gt_step']),
            sum(result['navigation_steps']),
            all(result['oracle_successes']),
            sum(result['navigation_errors'])/len(result['navigation_errors']) if len(result['navigation_errors']) > 0 else 0,
            result['successes'],
            result['gt_step'],
            result['gt_path'],
            result['navigation_errors']
            )
        metrics[config["Robot"]].add_sample(
            all(result['successes']),
            sum(result['gt_step']),
            sum(result['navigation_steps']),
            all(result['oracle_successes']),
            sum(result['navigation_errors'])/len(result['navigation_errors']) if len(result['navigation_errors']) > 0 else 0,
            result['successes'],
            result['gt_step'],
            result['gt_path'],
            result['navigation_errors']
            )
        metrics['result'].add_sample(
            all(result['successes']),
            sum(result['gt_step']),
            sum(result['navigation_steps']),
            all(result['oracle_successes']),
            sum(result['navigation_errors'])/len(result['navigation_errors']) if len(result['navigation_errors']) > 0 else 0,
            result['successes'],
            result['gt_step'],
            result['gt_path'],
            result['navigation_errors']
                )

        for i, step_config in enumerate(step_configs):
            logger.info(f"   >>> Running ST Sub-Task {i}: {step_config['Task instruction'][:70]}...")
            logger.info(step_config)
            agent = HabitatAgent(args, step_config, nav_model)
            result = agent.validate(step_task=True)
            metrics['step'].add_sample(
                all(result['successes']),
                sum(result['gt_step']),
                sum(result['navigation_steps']),
                all(result['oracle_successes']),
                sum(result['navigation_errors'])/len(result['navigation_errors']) if len(result['navigation_errors']) > 0 else 0,
                result['successes'],
                result['gt_step'],
                result['gt_path'],
                result['navigation_errors']
                    )

        # for key, metric in metrics.items():
        #     computed_metrics = metric.compute()
        #     logger.info(f"Type: {key}")
        #     for metric_name, value in computed_metrics.items():
        #         logger.info(f"  {metric_name}: {value:.4f}")

        verbose_dict = dict(
            step=step,
        )
        pbar.set_postfix(verbose_dict)
        pbar.update()
    

def main():
    args, global_cfg, logger, device_id = read_args()
    random_seed(args.seed)

    if args.tensorboard:
        writer_epoch = SummaryWriter(log_dir=args.tensorboard_path, filename_suffix='epoch')
    else:
        writer_epoch = None
    
    def custom_collate_fn(batch):
        if args.batch_size == 1:
            return batch[0]
        else:
            return batch

    if args.model_name == 'LLM Model':
        nav_model = ContinuousNav(args, global_cfg, logger, device_id)
    else:
        nav_model = RandomAgent()

    criterion = nn.CrossEntropyLoss(ignore_index=args.ignoreid, reduction='sum')
    # we save the best checkpoint for the best CSR
    CSR = 0.
    best_checkpoint = args.best_checkpoint if args.mode == 'test' else None 

    if args.mode == 'test':
        if args.episode_data:
            train_dataset = EpisodeDataset(args, mode='train')
            val_dataset = EpisodeDataset(args, mode='valid')
            test_dataset = EpisodeDataset(args, mode='test') 

        else:
            train_dataset = TaskDataset(args, mode='train')
            val_dataset = TaskDataset(args, mode='valid')
            test_dataset = TaskDataset(args, mode='test')
        print(f"DEBUG: Total tasks in test_dataset: {len(test_dataset.tasks)}")
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1, collate_fn=custom_collate_fn)

    else:
        if args.episode_data:
            train_dataset = EpisodeDataset(args, mode='train')
            val_dataset = EpisodeDataset(args, mode='valid')
            test_dataset = EpisodeDataset(args, mode='test') 
            
            # For unseen Test set
            # if args.split_by_scene:
            #     train_dataset, val_dataset, test_dataset = create_split_datasets([train_dataset, val_dataset, test_dataset], args)    
        else:
            train_dataset = TaskDataset(args, mode='train')
            val_dataset = TaskDataset(args, mode='valid')
            test_dataset = TaskDataset(args, mode='test')
            # For unseen Test set
            if args.split_by_scene:
                train_dataset, val_dataset, test_dataset = create_split_datasets([train_dataset, val_dataset, test_dataset], args)

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=custom_collate_fn)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=custom_collate_fn)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=custom_collate_fn)

    if args.mode == 'train':
        # metrics for diffrent tasks
        train_metrics = {
            'result': NavigationMetrics(),
            '1': NavigationMetrics(), #added this for rxr dataset
            '2': NavigationMetrics(),
            '3': NavigationMetrics(),
            '4': NavigationMetrics(),
            'spot': NavigationMetrics(),
            'stretch': NavigationMetrics(),
        }
        val_metrics = {
            'result': NavigationMetrics(),
            '1': NavigationMetrics(), #added this for rxr dataset
            '2': NavigationMetrics(),
            '3': NavigationMetrics(),
            '4': NavigationMetrics(),
            'spot': NavigationMetrics(),
            'stretch': NavigationMetrics(),
            'step': NavigationMetrics(),
        }
        for epoch in range(args.num_epochs):
            # train with imitation learning and dagger training
            lh_loss, st_loss = train_one_epoch(
                args, 
                epoch, 
                train_metrics, 
                train_dataloader, 
                nav_model.optimizer, 
                nav_model.lr_scheduler, 
                criterion, 
                nav_model,
                logger,
                args.resume_from_step
                )
            
            logger.info(f"###### Epoch: {epoch} ######")
            logger.info("###### Training ######")
            logger.info(f"Imiation learning Loss: {lh_loss:.4f}, dagger training Loss: {st_loss:.4f}")
            for key, metrics in train_metrics.items():
                computed_metrics = metrics.compute()
                logger.info(f"Type: {key}")
                for metric_name, value in computed_metrics.items():
                    logger.info(f"  {metric_name}: {value:.4f}")

            if epoch % args.save_ckpt_per_epochs == 0:
                checkpoint_path = args.save_checkpoints + '/' + 'epoch_' + str(epoch) + '.pth'
                save_checkpoint(nav_model.model, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")

            if writer_epoch:
                writer_epoch.add_scalars('Training metric', {
                    "SR": train_metrics['result'].compute()['success_rate'],
                    "OSR": train_metrics['result'].compute()['oracle_success_rate'],
                    "SPL": train_metrics['result'].compute()['spl'],
                    "NE": train_metrics['result'].compute()['navigation_error'],
                    "ISR": train_metrics['result'].compute()['independent_success_rate'],
                    "CSR": train_metrics['result'].compute()['conditional_success_rate'],
                    "CPL": train_metrics['result'].compute()['conditional_path_length'],
                }, epoch)

            validate_one_epoch(args, epoch, val_metrics, val_dataloader, nav_model, logger)
            logger.info("###### Validation ######")
            for key, metrics in val_metrics.items():
                computed_metrics = metrics.compute()
                logger.info(f"Type: {key}")
                for metric_name, value in computed_metrics.items():
                    logger.info(f"  {metric_name}: {value:.4f}")
                    if metric_name == 'Conditional Success Rate' and value > CSR:
                        CSR = value
                        best_checkpoint = checkpoint_path
                        logger.info(f"Best CSR is {CSR:.4f}, Best checkpoint updated to {best_checkpoint}")
            logger.info(f"Best CSR: {CSR:.4f}")

            if writer_epoch:
                writer_epoch.add_scalars('Validation metric', {
                    "SR": val_metrics['result'].compute()['success_rate'],
                    "OSR": val_metrics['result'].compute()['oracle_success_rate'],
                    "SPL": val_metrics['result'].compute()['spl'],
                    "NE": val_metrics['result'].compute()['navigation_error'],
                    "ISR": val_metrics['result'].compute()['independent_success_rate'],
                    "CSR": val_metrics['result'].compute()['conditional_success_rate'],
                    "CPL": val_metrics['result'].compute()['conditional_path_length'],
                }, epoch)

    if args.load_checkpoint:
        print(f'Loading checkpoint YANG INIIIIIIIIIIIIIIIIIIIIII from {best_checkpoint}')
        checkpoint = torch.load(best_checkpoint)
        if best_checkpoint is None:
            logger.error("No best_checkpoint found! Skipping loading logic.")
            # Either set a default or exit
            best_checkpoint = args.resume_from_checkpoint
        model_state_dict = nav_model.model.state_dict()
        # nav_model.model.load_state_dict(checkpoint['model_state_dict'])
        state_disk = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()}
        update_model_state = {}
        for key, val in state_disk.items():
            if key in model_state_dict and model_state_dict[key].shape == val.shape:
                update_model_state[key] = val
            else:
                logger.info(
                    'Ignore weight %s: %s' % (key, str(val.shape))
                )
        msg = nav_model.model.load_state_dict(update_model_state, strict=False)
        logger.info(f"Loaded best checkpoint from {best_checkpoint}")

    test_metrics = {
        'result': NavigationMetrics(),
        '1': NavigationMetrics(), #added this for rxr dataset
        '2': NavigationMetrics(),
        '3': NavigationMetrics(),
        '4': NavigationMetrics(),
        'spot': NavigationMetrics(),
        'stretch': NavigationMetrics(),
        'step': NavigationMetrics(),
    }

    validate_one_epoch(args, 0, test_metrics, test_dataloader, nav_model, logger)
    logger.info("###### Test ######")
    for key, metrics in test_metrics.items():
        computed_metrics = metrics.compute()
        logger.info(f"Type: {key}")
        for metric_name, value in computed_metrics.items():
            logger.info(f"  {metric_name}: {value:.4f}")
    
    if writer_epoch:
        writer_epoch.close()


if __name__ == '__main__':
    main()
# torchrun --nnodes=1 --nproc_per_node=4 train.py    
