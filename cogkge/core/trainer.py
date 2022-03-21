################################################################以下为最初版本###########################################################
#
# class Trainer(object):
#     def __init__(self,
#                  train_dataset,
#                  train_sampler,
#                  trainer_batch_size,
#                  model,
#                  loss,
#                  optimizer,
#                  negative_sampler,
#                  epoch,
#                  device,
#                  output_path,
#                  valid_dataset=None,
#                  valid_sampler=None,
#                  lookuptable_E=None,
#                  lookuptable_R=None,
#                  metric=None,
#                  lr_scheduler=None,
#                  log=None,
#                  load_checkpoint=None,
#                  visualization=False,
#                  apex=False,
#                  dataloaderX=False,
#                  num_workers=None,
#                  pin_memory=False,
#                  metric_step=None,
#                  save_step=None,
#                  metric_final_model=True,
#                  save_final_model=True,
#                  ):
#         self.train_dataset = train_dataset
#         self.train_sampler = train_sampler
#         self.trainer_batch_size = trainer_batch_size
#         self.model = model
#         self.loss = loss
#         self.optimizer = optimizer
#         self.negative_sampler = negative_sampler
#         self.epoch = epoch
#         self.device = device
#         self.valid_dataset = valid_dataset
#         self.valid_sampler = valid_sampler
#         self.lookuptable_E = lookuptable_E
#         self.lookuptable_R = lookuptable_R
#         self.metric = metric
#         self.lr_scheduler = lr_scheduler
#         self.log = log
#         self.load_checkpoint = load_checkpoint
#         self.visualization = visualization
#         self.apex = apex
#         self.dataloaderX = dataloaderX
#         self.num_workers = num_workers
#         self.pin_memory = pin_memory
#         self.metric_step = metric_step
#         self.save_step = save_step
#         self.metric_final_model = metric_final_model
#         self.save_final_model = save_final_model
#
#         self.data_name = train_dataset.data_name
#
#         self.visual_num = 100
#         if self.lookuptable_E.type is not None:
#             self.visual_type = self.lookuptable_E.type[:self.visual_num].numpy()
#         else:
#             self.visual_type = np.arctan2(np.random.normal(0, 2, self.visual_num),
#                                           np.random.normal(0, 2, self.visual_num))
#
#         # Set output_path
#         output_path = os.path.join(output_path, self.data_name)
#         self.output_path = cal_output_path(output_path, self.model.name)
#         self.output_path = self.output_path + "--{}epochs".format(self.epoch)
#         if not os.path.exists(self.output_path):
#             os.makedirs(self.output_path)
#
#         # Set logger
#         if log:
#             logger = save_logger(os.path.join(self.output_path, "trainer_run.log"))
#             logger.info("Data Experiment Output Path:{}".format(self.output_path))
#             self.logger = logger
#
#         # Load Apex
#         if self.apex:
#             if "apex" not in sys.modules:
#                 logger.info("Apex has not been installed!Force the parameter to be False.")
#                 self.apex = False
#             else:
#                 from apex import amp
#                 self.model, self.optimizer = amp.initialize(self.model.to(self.device), self.optimizer, opt_level="O1")
#
#         # Load Data
#         if self.dataloaderX:
#             self.train_loader = DataLoaderX(dataset=self.train_dataset, sampler=self.train_sampler,
#                                             batch_size=self.trainer_batch_size, num_workers=self.num_workers,
#                                             pin_memory=self.pin_memory)
#             if self.valid_dataset:
#                 self.valid_loader = DataLoaderX(dataset=self.valid_dataset, sampler=self.valid_sampler,
#                                                 batch_size=self.trainer_batch_size, num_workers=self.num_workers,
#                                                 pin_memory=self.pin_memory)
#         else:
#             self.train_loader = Data.DataLoader(dataset=self.train_dataset, sampler=self.train_sampler,
#                                                 batch_size=self.trainer_batch_size, num_workers=self.num_workers,
#                                                 pin_memory=self.pin_memory)
#             if self.valid_dataset:
#                 self.valid_loader = Data.DataLoader(dataset=self.valid_dataset, sampler=self.valid_sampler,
#                                                     batch_size=self.trainer_batch_size, num_workers=self.num_workers,
#                                                     pin_memory=self.pin_memory)
#
#         # Load Checkpoint
#         self.trained_epoch = 0
#         if self.load_checkpoint:
#             if os.path.exists(self.load_checkpoint):
#                 string = self.load_checkpoint
#                 pattern = r"^.*?/checkpoints/.*?_(.*?)epochs$"
#                 match = re.search(pattern, string)
#                 self.trained_epoch = int(match.group(1))
#                 self.model.load_state_dict(torch.load(os.path.join(self.load_checkpoint, "Model.pkl")))
#                 self.optimizer.load_state_dict(torch.load(os.path.join(self.load_checkpoint, "Optimizer.pkl")))
#                 self.lr_scheduler.load_state_dict(torch.load(os.path.join(self.load_checkpoint, "Lr_Scheduler.pkl")))
#             else:
#                 raise FileExistsError("Checkpoint path doesn't exist!")
#
#
#         print("Available cuda devices:", torch.cuda.device_count())
#         self.parallel_model = torch.nn.DataParallel(self.model)
#         self.parallel_model = self.parallel_model.to(self.device)
#
#         # Load Visualization
#         self.writer = None
#         if self.visualization == True:
#             self.visualization_path = os.path.join(self.output_path, "visualization", self.model.name)
#             if not os.path.exists(self.visualization_path):
#                 os.makedirs(self.visualization_path)
#             self.writer = SummaryWriter(self.visualization_path)
#             self.logger.info(
#                 "The visualization path is" + self.visualization_path)
#
#         # Load Metric
#         if self.metric:
#             self.metric.initialize(device=self.device,
#                                    total_epoch=self.epoch,
#                                    metric_type="valid",
#                                    node_dict_len=len(self.lookuptable_E),
#                                    model_name=self.model.name,
#                                    logger=self.logger,
#                                    writer=self.writer,
#                                    train_dataset=self.train_dataset,
#                                    valid_dataset=self.valid_dataset)
#             if self.metric.link_prediction_filt:
#                 self.metric.establish_correct_triplets_dict()
#
#     def train(self):
#         if self.epoch - self.trained_epoch <= 0:
#             raise ValueError("Trained_epoch is bigger than total_epoch!")
#
#         for epoch in range(self.epoch - self.trained_epoch):
#             current_epoch = epoch + 1 + self.trained_epoch
#
#             # Training Progress
#             train_epoch_loss = 0.0
#             for train_step, train_positive in enumerate(tqdm(self.train_loader)):
#                 train_positive = train_positive.to(self.device)
#                 train_negative = self.negative_sampler.create_negative(train_positive[:, :3])
#                 train_positive_score = self.parallel_model(train_positive)
#                 if len(train_positive[0]) == 5:
#                     train_negative = torch.cat((train_negative, train_positive[:, 3:]), dim=1)
#                 train_negative_score = self.parallel_model(train_negative)
#                 penalty = self.model.get_penalty() if hasattr(self.model, 'get_penalty') else 0
#                 train_loss = self.loss(train_positive_score, train_negative_score, penalty)
#
#                 self.optimizer.zero_grad()
#                 if self.apex:
#                     from apex import amp
#                     with amp.scale_loss(train_loss, self.optimizer) as scaled_loss:
#                         scaled_loss.backward()
#                 else:
#                     train_loss.backward()
#                 train_epoch_loss = train_epoch_loss + train_loss.item()
#                 self.optimizer.step()
#
#             valid_epoch_loss = 0.0
#             with torch.no_grad():
#                 for valid_step, valid_positive in enumerate(self.valid_loader):
#                     valid_positive = valid_positive.to(self.device)
#                     valid_negative = self.negative_sampler.create_negative(valid_positive[:, :3])
#                     valid_positive_score = self.parallel_model(valid_positive)
#                     if len(valid_positive[0]) == 5:
#                         valid_negative = torch.cat((valid_negative, valid_positive[:, 3:]), dim=1)
#                     valid_negative_score = self.parallel_model(valid_negative)
#                     penalty = self.model.get_penalty() if hasattr(self.model, 'get_penalty') else 0
#                     valid_loss = self.loss(valid_positive_score, valid_negative_score, penalty)
#                     valid_epoch_loss = valid_epoch_loss + valid_loss.item()
#
#             print("Epoch{}/{}   Train Loss:".format(current_epoch, self.epoch), train_epoch_loss / (train_step + 1),
#                   " Valid Loss:", valid_epoch_loss / (valid_step + 1))
#
#             # Metric Progress
#             if self.metric_step and (current_epoch) % self.metric_step == 0 or self.metric_final_model and (
#                     current_epoch) == self.epoch:
#                 print("Evaluating Model {} on Valid Dataset...".format(self.model.name))
#                 self.metric.caculate(model=self.parallel_model, current_epoch=current_epoch)
#                 self.metric.print_current_table()
#                 self.metric.log()
#                 self.metric.write()
#                 print("-----------------------------------------------------------------------")
#
#                 # Scheduler Progress
#                 self.lr_scheduler.step(self.metric.get_Raw_MR())
#
#             # Visualization Process
#             if self.visualization:
#                 self.writer.add_scalars("Loss", {"train_loss": train_epoch_loss,
#                                                  "valid_loss": valid_epoch_loss}, current_epoch)
#                 # if self.model.entity_embedding is not None:
#                 #     embedding = self.model.entity_embedding.weight.data.clone().cpu().numpy()[:self.visual_num]
#                 #     embedding = TSNE(negative_gradient_method="bh").fit(embedding)
#                 #     plt.scatter(embedding[:, 0], embedding[:, 1], c=self.visual_type)
#                 #     plt.show()
#                 # if epoch == 0:
#                 #     fake_data = torch.zeros(self.trainer_batch_size, 3).long()
#                 #     self.writer.add_graph(self.model.cpu(), fake_data)
#                 #     self.model.to(self.device)
#
#             # Save Checkpoint and Final Model Process
#             if (self.save_step and (current_epoch) % self.save_step == 0) or (current_epoch) == self.epoch:
#                 if not os.path.exists(os.path.join(self.output_path, "checkpoints",
#                                                    "{}_{}epochs".format(self.model.name, current_epoch))):
#                     os.makedirs(os.path.join(self.output_path, "checkpoints",
#                                              "{}_{}epochs".format(self.model.name, current_epoch)))
#                     self.logger.info(os.path.join(self.output_path, "checkpoints",
#                                                   "{}_{}epochs ".format(self.model.name,
#                                                                         current_epoch)) + 'created successfully!')
#                 torch.save(self.model.state_dict(), os.path.join(self.output_path, "checkpoints",
#                                                                  "{}_{}epochs".format(self.model.name, current_epoch),
#                                                                  "Model.pkl"))
#                 torch.save(self.optimizer.state_dict(), os.path.join(self.output_path, "checkpoints",
#                                                                      "{}_{}epochs".format(self.model.name,
#                                                                                           current_epoch),
#                                                                      "Optimizer.pkl"))
#                 torch.save(self.lr_scheduler.state_dict(), os.path.join(self.output_path, "checkpoints",
#                                                                         "{}_{}epochs".format(self.model.name,
#                                                                                              current_epoch),
#                                                                         "Lr_Scheduler.pkl"))
#                 self.logger.info(os.path.join(self.output_path, "checkpoints", "{}_{}epochs ".format(self.model.name,
#                                                                                                      current_epoch)) + "saved successfully")
#
#         # Show Best Metric Result
#         if self.metric_step:
#             self.metric.print_best_table(front=5, key="Filt_Hits@10")
################################################################以下继承两个类的版本###########################################################
# class BaseTrainer(object):
#     def __init__(self,
#                  train_dataset,
#                  train_sampler,
#                  trainer_batch_size,
#                  model,
#                  loss,
#                  optimizer,
#                  negative_sampler,
#                  epoch,
#                  device,
#                  output_path,
#                  valid_dataset=None,
#                  valid_sampler=None,
#                  lookuptable_E=None,
#                  lookuptable_R=None,
#                  metric=None,
#                  lr_scheduler=None,
#                  log=None,
#                  load_checkpoint=None,
#                  visualization=False,
#                  apex=False,
#                  dataloaderX=False,
#                  num_workers=None,
#                  pin_memory=False,
#                  metric_step=None,
#                  save_step=None,
#                  metric_final_model=True,
#                  save_final_model=True,
#                  ):
#         self.train_dataset = train_dataset
#         self.train_sampler = train_sampler
#         self.trainer_batch_size = trainer_batch_size
#         self.model = model
#         self.loss = loss
#         self.optimizer = optimizer
#         self.negative_sampler = negative_sampler
#         self.epoch = epoch
#         self.device = device
#         self.valid_dataset = valid_dataset
#         self.valid_sampler = valid_sampler
#         self.lookuptable_E = lookuptable_E
#         self.lookuptable_R = lookuptable_R
#         self.metric = metric
#         self.lr_scheduler = lr_scheduler
#         self.log = log
#         self.load_checkpoint = load_checkpoint
#         self.visualization = visualization
#         self.apex = apex
#         self.dataloaderX = dataloaderX
#         self.num_workers = num_workers
#         self.pin_memory = pin_memory
#         self.metric_step = metric_step
#         self.save_step = save_step
#         self.metric_final_model = metric_final_model
#         self.save_final_model = save_final_model
#
#         self.data_name = train_dataset.data_name
#
#         self.visual_num = 100
#         if self.lookuptable_E.type is not None:
#             self.visual_type = self.lookuptable_E.type[:self.visual_num].numpy()
#         else:
#             self.visual_type = np.arctan2(np.random.normal(0, 2, self.visual_num),
#                                           np.random.normal(0, 2, self.visual_num))
#
#         # Set output_path
#         output_path = os.path.join(output_path, self.data_name)
#         self.output_path = cal_output_path(output_path, self.model.name)
#         self.output_path = self.output_path + "--{}epochs".format(self.epoch)
#         if not os.path.exists(self.output_path):
#             os.makedirs(self.output_path)
#
#         # Set logger
#         if log:
#             logger = save_logger(os.path.join(self.output_path, "trainer_run.log"))
#             logger.info("Data Experiment Output Path:{}".format(self.output_path))
#             self.logger = logger
#
#         # Load Apex
#         if self.apex:
#             if "apex" not in sys.modules:
#                 logger.info("Apex has not been installed!Force the parameter to be False.")
#                 self.apex = False
#             else:
#                 from apex import amp
#                 self.model, self.optimizer = amp.initialize(self.model.to(self.device), self.optimizer, opt_level="O1")
#
#         # Load Data
#         if self.dataloaderX:
#             self.train_loader = DataLoaderX(dataset=self.train_dataset, sampler=self.train_sampler,
#                                             batch_size=self.trainer_batch_size, num_workers=self.num_workers,
#                                             pin_memory=self.pin_memory)
#             if self.valid_dataset:
#                 self.valid_loader = DataLoaderX(dataset=self.valid_dataset, sampler=self.valid_sampler,
#                                                 batch_size=self.trainer_batch_size, num_workers=self.num_workers,
#                                                 pin_memory=self.pin_memory)
#         else:
#             self.train_loader = Data.DataLoader(dataset=self.train_dataset, sampler=self.train_sampler,
#                                                 batch_size=self.trainer_batch_size, num_workers=self.num_workers,
#                                                 pin_memory=self.pin_memory)
#             if self.valid_dataset:
#                 self.valid_loader = Data.DataLoader(dataset=self.valid_dataset, sampler=self.valid_sampler,
#                                                     batch_size=self.trainer_batch_size, num_workers=self.num_workers,
#                                                     pin_memory=self.pin_memory)
#
#         # Load Checkpoint
#         self.trained_epoch = 0
#         if self.load_checkpoint:
#             if os.path.exists(self.load_checkpoint):
#                 string = self.load_checkpoint
#                 pattern = r"^.*?/checkpoints/.*?_(.*?)epochs$"
#                 match = re.search(pattern, string)
#                 self.trained_epoch = int(match.group(1))
#                 self.model.load_state_dict(torch.load(os.path.join(self.load_checkpoint, "Model.pkl")))
#                 self.optimizer.load_state_dict(torch.load(os.path.join(self.load_checkpoint, "Optimizer.pkl")))
#                 self.lr_scheduler.load_state_dict(torch.load(os.path.join(self.load_checkpoint, "Lr_Scheduler.pkl")))
#             else:
#                 raise FileExistsError("Checkpoint path doesn't exist!")
#
#
#         print("Available cuda devices:", torch.cuda.device_count())
#         self.parallel_model = torch.nn.DataParallel(self.model)
#         self.parallel_model = self.parallel_model.to(self.device)
#
#         # Load Visualization
#         self.writer = None
#         if self.visualization == True:
#             self.visualization_path = os.path.join(self.output_path, "visualization", self.model.name)
#             if not os.path.exists(self.visualization_path):
#                 os.makedirs(self.visualization_path)
#             self.writer = SummaryWriter(self.visualization_path)
#             self.logger.info(
#                 "The visualization path is" + self.visualization_path)
#
#         # Load Metric
#         if self.metric:
#             self.metric.initialize(device=self.device,
#                                    total_epoch=self.epoch,
#                                    metric_type="valid",
#                                    node_dict_len=len(self.lookuptable_E),
#                                    model_name=self.model.name,
#                                    logger=self.logger,
#                                    writer=self.writer,
#                                    train_dataset=self.train_dataset,
#                                    valid_dataset=self.valid_dataset)
#             if self.metric.link_prediction_filt:
#                 self.metric.establish_correct_triplets_dict()
#
#     def get_loss_strategy(self,data_positive):
#         loss=0
#         return loss
#
#     def train(self):
#         if self.epoch - self.trained_epoch <= 0:
#             raise ValueError("Trained_epoch is bigger than total_epoch!")
#
#         for epoch in range(self.epoch - self.trained_epoch):
#             current_epoch = epoch + 1 + self.trained_epoch
#
#             # Training Progress
#             train_epoch_loss = 0.0
#             for train_step, train_positive in enumerate(tqdm(self.train_loader)):
#
#                 train_loss=self.get_loss_strategy(train_positive)
#
#                 self.optimizer.zero_grad()
#                 if self.apex:
#                     from apex import amp
#                     with amp.scale_loss(train_loss, self.optimizer) as scaled_loss:
#                         scaled_loss.backward()
#                 else:
#                     train_loss.backward()
#                 train_epoch_loss = train_epoch_loss + train_loss.item()
#                 self.optimizer.step()
#
#             valid_epoch_loss = 0.0
#             with torch.no_grad():
#                 for valid_step, valid_positive in enumerate(self.valid_loader):
#                     valid_loss = self.get_loss_strategy(valid_positive)
#                     valid_epoch_loss = valid_epoch_loss + valid_loss.item()
#
#             print("Epoch{}/{}   Train Loss:".format(current_epoch, self.epoch), train_epoch_loss / (train_step + 1),
#                   " Valid Loss:", valid_epoch_loss / (valid_step + 1))
#
#             # Metric Progress
#             if self.metric_step and (current_epoch) % self.metric_step == 0 or self.metric_final_model and (
#                     current_epoch) == self.epoch:
#                 print("Evaluating Model {} on Valid Dataset...".format(self.model.name))
#                 self.metric.caculate(model=self.parallel_model, current_epoch=current_epoch)
#                 self.metric.print_current_table()
#                 self.metric.log()
#                 self.metric.write()
#                 print("-----------------------------------------------------------------------")
#
#                 # Scheduler Progress
#                 self.lr_scheduler.step(self.metric.get_Raw_MR())
#
#             # Visualization Process
#             if self.visualization:
#                 self.writer.add_scalars("Loss", {"train_loss": train_epoch_loss,
#                                                  "valid_loss": valid_epoch_loss}, current_epoch)
#                 if hasattr(self.model, 'entity_embedding'):
#                 # if self.model.entity_embedding is not None:
#                     embedding = self.model.entity_embedding.weight.data.clone().cpu().numpy()[:self.visual_num]
#                     embedding = TSNE(negative_gradient_method="bh").fit(embedding)
#                     plt.scatter(embedding[:, 0], embedding[:, 1], c=self.visual_type)
#                     plt.show()
#                 if epoch == 0:
#                     fake_data = torch.zeros(self.trainer_batch_size, 3).long()
#                     self.writer.add_graph(self.model.cpu(), fake_data)
#                     self.model.to(self.device)
#
#             # Save Checkpoint and Final Model Process
#             if (self.save_step and (current_epoch) % self.save_step == 0) or (current_epoch) == self.epoch:
#                 if not os.path.exists(os.path.join(self.output_path, "checkpoints",
#                                                    "{}_{}epochs".format(self.model.name, current_epoch))):
#                     os.makedirs(os.path.join(self.output_path, "checkpoints",
#                                              "{}_{}epochs".format(self.model.name, current_epoch)))
#                     self.logger.info(os.path.join(self.output_path, "checkpoints",
#                                                   "{}_{}epochs ".format(self.model.name,
#                                                                         current_epoch)) + 'created successfully!')
#                 torch.save(self.model.state_dict(), os.path.join(self.output_path, "checkpoints",
#                                                                  "{}_{}epochs".format(self.model.name, current_epoch),
#                                                                  "Model.pkl"))
#                 torch.save(self.optimizer.state_dict(), os.path.join(self.output_path, "checkpoints",
#                                                                      "{}_{}epochs".format(self.model.name,
#                                                                                           current_epoch),
#                                                                      "Optimizer.pkl"))
#                 torch.save(self.lr_scheduler.state_dict(), os.path.join(self.output_path, "checkpoints",
#                                                                         "{}_{}epochs".format(self.model.name,
#                                                                                              current_epoch),
#                                                                         "Lr_Scheduler.pkl"))
#                 self.logger.info(os.path.join(self.output_path, "checkpoints", "{}_{}epochs ".format(self.model.name,
#                                                                                                      current_epoch)) + "saved successfully")
#
#         # Show Best Metric Result
#         if self.metric_step:
#             self.metric.print_best_table(front=5, key="Filt_Hits@10")
#
# class ClassifyTrainer(BaseTrainer):
#     def __init__(self,
#                  train_dataset,
#                  train_sampler,
#                  trainer_batch_size,
#                  model,
#                  loss,
#                  optimizer,
#                  negative_sampler,
#                  epoch,
#                  device,
#                  output_path,
#                  valid_dataset=None,
#                  valid_sampler=None,
#                  lookuptable_E=None,
#                  lookuptable_R=None,
#                  metric=None,
#                  lr_scheduler=None,
#                  log=None,
#                  load_checkpoint=None,
#                  visualization=False,
#                  apex=False,
#                  dataloaderX=False,
#                  num_workers=None,
#                  pin_memory=False,
#                  metric_step=None,
#                  save_step=None,
#                  metric_final_model=True,
#                  save_final_model=True,
#                  ):
#         super().__init__(train_dataset=train_dataset,
#                          train_sampler=train_sampler,
#                          trainer_batch_size=trainer_batch_size,
#                          model=model,
#                          loss=loss,
#                          optimizer=optimizer,
#                          negative_sampler=negative_sampler,
#                          epoch=epoch,
#                          device=device,
#                          output_path=output_path,
#                          valid_dataset=valid_dataset,
#                          valid_sampler=valid_sampler,
#                          lookuptable_E=lookuptable_E,
#                          lookuptable_R=lookuptable_R,
#                          metric=metric,
#                          lr_scheduler=lr_scheduler,
#                          log=log,
#                          load_checkpoint=load_checkpoint,
#                          visualization=visualization,
#                          apex=apex,
#                          dataloaderX=dataloaderX,
#                          num_workers=num_workers,
#                          pin_memory=pin_memory,
#                          metric_step=metric_step,
#                          save_step=save_step,
#                          metric_final_model=metric_final_model,
#                          save_final_model=save_final_model)
#
#     def get_loss_strategy(self,data_positive):
#         h_r_true = data_positive[0][:,:2].long().to(self.device)
#         t_true=data_positive[1].float().to(self.device)
#         data_positive_probability = self.parallel_model(h_r_true)
#         if hasattr(self.model, 'get_sampel_label_index'):
#             sample_index=self.model.get_sampel_label_index(data_positive[0].shape[0])
#         else:
#             sample_index=torch.arange(0,data_positive[0].shape[0], step = 1)
#         data_loss = self.loss(data_positive_probability,t_true[sample_index])
#         return data_loss
#
#
#
# class ScoreTrainer(BaseTrainer):
#     def __init__(self,
#                  train_dataset,
#                  train_sampler,
#                  trainer_batch_size,
#                  model,
#                  loss,
#                  optimizer,
#                  negative_sampler,
#                  epoch,
#                  device,
#                  output_path,
#                  valid_dataset=None,
#                  valid_sampler=None,
#                  lookuptable_E=None,
#                  lookuptable_R=None,
#                  metric=None,
#                  lr_scheduler=None,
#                  log=None,
#                  load_checkpoint=None,
#                  visualization=False,
#                  apex=False,
#                  dataloaderX=False,
#                  num_workers=None,
#                  pin_memory=False,
#                  metric_step=None,
#                  save_step=None,
#                  metric_final_model=True,
#                  save_final_model=True,
#                  ):
#         super().__init__(train_dataset=train_dataset,
#                          train_sampler=train_sampler,
#                          trainer_batch_size=trainer_batch_size,
#                          model=model,
#                          loss=loss,
#                          optimizer=optimizer,
#                          negative_sampler=negative_sampler,
#                          epoch=epoch,
#                          device=device,
#                          output_path=output_path,
#                          valid_dataset=valid_dataset,
#                          valid_sampler=valid_sampler,
#                          lookuptable_E=lookuptable_E,
#                          lookuptable_R=lookuptable_R,
#                          metric=metric,
#                          lr_scheduler=lr_scheduler,
#                          log=log,
#                          load_checkpoint=load_checkpoint,
#                          visualization=visualization,
#                          apex=apex,
#                          dataloaderX=dataloaderX,
#                          num_workers=num_workers,
#                          pin_memory=pin_memory,
#                          metric_step=metric_step,
#                          save_step=save_step,
#                          metric_final_model=metric_final_model,
#                          save_final_model=save_final_model)
#
#     def get_loss_strategy(self,data_positive):
#         data_positive = data_positive.to(self.device)
#         data_negative = self.negative_sampler.create_negative(data_positive[:, :3])
#         data_positive_score = self.parallel_model(data_positive)
#         if len(data_positive[0]) == 5:
#             data_negative = torch.cat((data_negative, data_positive[:, 3:]), dim=1)
#         data_negative_score = self.parallel_model(data_negative)
#         penalty = self.model.get_penalty() if hasattr(self.model, 'get_penalty') else 0
#         data_loss = self.loss(data_positive_score, data_negative_score, penalty)
#         return data_loss
##################################################以下为最新版本的trainer###################################################
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as Data
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from cogkge.core import DataLoaderX
from .log import save_logger
from ..utils.kr_utils import cal_output_path

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def reduce_mean(tensor,nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt,op=dist.ReduceOp.SUM)
    rt = rt/nprocs
    return rt

class Trainer(object):
    def __init__(self,
                 train_dataset,
                 train_sampler,
                 valid_dataset,
                 valid_sampler,
                 trainer_batch_size,
                 model,
                 loss,
                 metric,
                 negative_sampler,
                 optimizer,
                 total_epoch,
                 device,
                 output_path,
                 lookuptable_E=None,
                 lookuptable_R=None,
                 time_lut=None,
                 lr_scheduler=None,
                 apex=False,
                 dataloaderX=False,
                 num_workers=0,
                 pin_memory=True,
                 use_metric_epoch=0.1,
                 use_tensorboard_epoch=0.1,
                 use_savemodel_epoch=0.1,
                 use_matplotlib_epoch=0.1,
                 checkpoint_path=None,
                 rank=-1,
                 ):
        # 传入参数
        self.train_dataset = train_dataset
        self.train_sampler = train_sampler
        self.valid_dataset = valid_dataset
        self.valid_sampler = valid_sampler
        self.trainer_batch_size = trainer_batch_size
        self.model = model
        self.loss = loss
        self.metric = metric
        self.negative_sampler = negative_sampler
        self.optimizer = optimizer
        self.total_epoch = total_epoch
        self.device = device
        self.output_path = cal_output_path(os.path.join(output_path, train_dataset.data_name),
                                           self.model.model_name) + "--{}epochs".format(total_epoch)
        self.lookuptable_E = lookuptable_E
        self.lookuptable_R = lookuptable_R
        self.time_lut=time_lut
        self.lr_scheduler = lr_scheduler
        self.apex = apex
        self.dataloaderX = dataloaderX
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.use_metric_epoch = use_metric_epoch
        self.use_tensorboard_epoch = use_tensorboard_epoch
        self.use_savemodel_epoch = use_savemodel_epoch
        self.use_matplotlib_epoch = use_matplotlib_epoch
        self.checkpoint_path = checkpoint_path

        # 全局变量
        self.average_train_epoch_loss_list = []  # 平均训练损失
        self.average_valid_epoch_loss_list = []  # 平均验证损失
        self.current_epoch_list = []  # 目前轮数list
        self.writer = None  # tensorboard
        self.trained_epoch = 0  # 已经训练的轮数
        self.logger = None  # log
        self.visualization_path = os.path.join(self.output_path, "visualization", self.model.model_name)
        self.log_path = os.path.join(self.output_path, "trainer_run.log")
        self.data_name = train_dataset.data_name
        self.model_name = self.model.model_name
        self.rank = rank

        # Set path
        if self.rank in [-1,0]:
            if not os.path.exists(self.output_path):
                os.makedirs(self.output_path)
            if not os.path.exists(self.visualization_path):
                os.makedirs(self.visualization_path)

        # Set Model
        time_dict_len=0
        nodetype_dict_len=0
        relationtype_dict_len=0
        if hasattr(time_lut, "vocab") and time_lut.vocab is not None:
            time_dict_len = len(time_lut.vocab)
        if hasattr(lookuptable_E, "type") and lookuptable_E.type is not None:
            nodetype_dict_len = len(set(lookuptable_E.type.numpy()))
        if hasattr(lookuptable_R, "type") and lookuptable_R.type is not None:
            relationtype_dict_len = len(set(lookuptable_R.type.numpy()))

        # time_dict_len=len(time_lut.vocab) if hasattr(time_lut,"vocab") else 0
        # nodetype_dict_len = len(set(lookuptable_E.type.numpy()))if hasattr(lookuptable_E,"type") else 0
        # relationtype_dict_len = len(set(lookuptable_R.type.numpy())) if hasattr(lookuptable_R, "type") else 0
        self.model.set_model_config(model_loss=self.loss,
                                    model_metric=metric,
                                    model_negative_sampler=negative_sampler,
                                    model_device=self.device,
                                    time_dict_len=time_dict_len,
                                    nodetype_dict_len=nodetype_dict_len,
                                    relationtype_dict_len=relationtype_dict_len)
        if self.rank == -1:
            self.model = self.model.to(self.device)
        else:
            self.model = self.model.cuda(self.rank)
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = DDP(self.model,
                             device_ids=[self.rank],
                             output_device=self.rank,
                             find_unused_parameters=False,
                             broadcast_buffers=False)

        # Set Apex
        if self.apex:
            if "apex" not in sys.modules:
                # print("Please install apex!")
                self.apex = False
            else:
                from apex import amp
                self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level="O1")
        # Set log
        self.logger = save_logger(self.log_path,rank=self.rank)
        self.logger.info("Data Experiment Output Path:{}".format(os.path.abspath(self.output_path)))

        # Set Checkpoint
        if self.checkpoint_path != None:
            if os.path.exists(self.checkpoint_path):
                string = self.checkpoint_path
                pattern = r"^.*?/checkpoints/.*?_(.*?)epochs$"
                match = re.search(pattern, string)
                self.trained_epoch = int(match.group(1))
                self.model.load_state_dict(torch.load(os.path.join(self.checkpoint_path, "Model.pkl")))
                self.optimizer.load_state_dict(torch.load(os.path.join(self.checkpoint_path, "Optimizer.pkl")))
                self.lr_scheduler.load_state_dict(torch.load(os.path.join(self.checkpoint_path, "Lr_Scheduler.pkl")))
            else:
                raise FileExistsError("Checkpoint path doesn't exist!")

        # Set Tensorboard
        if use_tensorboard_epoch != 0.1:
            self.writer = SummaryWriter(self.visualization_path)

        # Set DataLoader
        if self.dataloaderX:
            self.train_loader = DataLoaderX(dataset=self.train_dataset, sampler=self.train_sampler,
                                            batch_size=self.trainer_batch_size, num_workers=self.num_workers,
                                            pin_memory=self.pin_memory)
            self.valid_loader = DataLoaderX(dataset=self.valid_dataset, sampler=self.valid_sampler,
                                            batch_size=self.trainer_batch_size, num_workers=self.num_workers,
                                                   pin_memory=self.pin_memory)
        else:
            self.train_loader = Data.DataLoader(dataset=self.train_dataset, sampler=self.train_sampler,
                                                batch_size=self.trainer_batch_size, num_workers=self.num_workers,
                                                pin_memory=self.pin_memory)
            self.valid_loader = Data.DataLoader(dataset=self.valid_dataset, sampler=self.valid_sampler,
                                                batch_size=self.trainer_batch_size, num_workers=self.num_workers,
                                                pin_memory=self.pin_memory)

        # Set Metric
        if self.metric:
            self.metric.initialize(device=self.device,
                                   total_epoch=self.total_epoch,
                                   metric_type="valid",
                                   node_dict_len=len(self.lookuptable_E),
                                   model_name=self.model_name,
                                   logger=self.logger,
                                   writer=self.writer,
                                   train_dataset=self.train_dataset,
                                   valid_dataset=self.valid_dataset)
            if self.metric.link_prediction_filt and self.rank in [-1,0]:
                self.metric.establish_correct_triplets_dict()

        # Set Multi GPU

    def train(self):
        if self.total_epoch <= self.trained_epoch:
            raise ValueError("Trained_epoch is bigger than total_epoch!")

        for epoch in range(self.total_epoch - self.trained_epoch):
            self.current_epoch = epoch + 1 + self.trained_epoch

            # Train Progress
            train_epoch_loss = 0.0
            if self.rank == -1:
                for train_step, batch in enumerate(tqdm(self.train_loader)):
                    train_loss = self.model.loss(batch)
                    train_epoch_loss += train_loss.item()
                    self.optimizer.zero_grad()
                    if self.apex:
                        from apex import amp
                        with amp.scale_loss(train_loss, self.optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        train_loss.backward()
                    self.optimizer.step()
            else:
                self.train_sampler.set_epoch(epoch)
                self.model.train()
                for train_step, batch in enumerate(self.train_loader):
                    train_loss = self.model.module.loss(batch)
                    train_epoch_loss += reduce_mean(train_loss,dist.get_world_size()).item()
                    self.optimizer.zero_grad()
                    if self.apex:
                        from apex import amp
                        with amp.scale_loss(train_loss, self.optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        train_loss.backward()
                    self.optimizer.step()

            # Valid Process
            if self.rank in [-1,0]:
                with torch.no_grad():
                    valid_epoch_loss = 0.0
                    valid_model = self.model if self.rank == -1 else self.model.module
                    valid_model.eval()
                    for batch in self.valid_loader:
                        valid_loss = valid_model.loss(batch)
                        valid_epoch_loss += valid_loss.item()
                        # break
                    average_train_epoch_loss = train_epoch_loss / len(self.train_dataset)
                    average_valid_epoch_loss = valid_epoch_loss / len(self.valid_dataset)
                    self.average_train_epoch_loss_list.append(average_train_epoch_loss)
                    self.average_valid_epoch_loss_list.append(average_valid_epoch_loss)
                    self.current_epoch_list.append(self.current_epoch)
                    self.logger.info("Epoch{}/{}   Train Loss: {}   Valid Loss: {}".format(self.current_epoch,
                                                                                self.total_epoch,
                                                                                average_train_epoch_loss,
                                                                                average_valid_epoch_loss))

            # Metric Progress
            if self.use_metric_epoch and self.current_epoch % self.use_metric_epoch == 0 and self.rank in [-1,0]:
                self.use_metric()
            # Tensorboard Process
            if self.current_epoch % self.use_tensorboard_epoch == 0:
                self.use_tensorboard(average_train_epoch_loss, average_valid_epoch_loss)
            # Savemodel Process
            if self.current_epoch % self.use_savemodel_epoch == 0:
                self.use_savemodel()
            # Matlotlib Process
            if self.current_epoch % self.use_matplotlib_epoch == 0:
                self.use_matplotlib()

    def use_metric(self):
        print("Evaluating Model {} on Valid Dataset...".format(self.model_name))
        self.metric.caculate(model=self.model, current_epoch=self.current_epoch)
        self.metric.print_current_table()
        self.metric.log()
        self.metric.write()
        print("-----------------------------------------------------------------------")

    def use_tensorboard(self, average_train_epoch_loss, average_valid_epoch_loss):
        self.writer.add_scalars("Loss", {"train_loss": average_train_epoch_loss,
                                         "valid_loss": average_valid_epoch_loss},
                                self.current_epoch)

    def use_savemodel(self):
        checkpoint_path = os.path.join(self.output_path, "checkpoints",
                                       "{}_{}epochs".format(self.model_name, self.current_epoch))
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        torch.save(self.model.state_dict(), os.path.join(checkpoint_path, "Model.pkl"))
        torch.save(self.optimizer.state_dict(), os.path.join(checkpoint_path, "Optimizer.pkl"))
        torch.save(self.lr_scheduler.state_dict(), os.path.join(checkpoint_path, "Lr_Scheduler.pkl"))

    def use_matplotlib(self):
        plt.figure()
        plt.plot(np.array(self.current_epoch_list), np.array(self.average_train_epoch_loss_list),
                 label="train_loss",
                 color="cornflowerblue",
                 linewidth=4, linestyle="-")
        plt.plot(np.array(self.current_epoch_list), np.array(self.average_valid_epoch_loss_list),
                 label="valid_loss",
                 color="darkviolet",
                 linewidth=4, linestyle="--")
        plt.yscale('log')
        plt.legend(loc="upper right")
        plt.show()
