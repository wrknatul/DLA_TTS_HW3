import typing as tp
import torch

import hw_tts.waveglow as waveglow
import hw_tts.synthesis.synthesis as synthesis
import hw_tts.synthesis.utils

from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from hw_tts.base import BaseTrainer
from hw_tts.utils import inf_loop, MetricTracker
from hw_tts.loss import FastSpeechLoss
from hw_tts.audio import hparams_audio


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            model,
            criterion,
            optimizer,
            config,
            device,
            dataloader,
            lr_scheduler=None,
            len_epoch=None,
            skip_oom=True,
    ):
        super().__init__(model, criterion, optimizer, config, device)
        self.skip_oom = skip_oom
        self.config = config
        self.train_dataloader = dataloader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        self.lr_scheduler = lr_scheduler
        self.log_step = 200

        self.train_metrics = MetricTracker(
            "loss", "mel_loss", "duration_loss", "pitch_loss", "energy_loss", "grad norm", writer=self.writer
        )

        self.test_data = hw_tts.synthesis.utils.get_data()

        self.loss = FastSpeechLoss()

        self.waveglow = waveglow.utils.get_WaveGlow().to(self.device)

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)
        step = 0
        for batchs_idx, batchs in enumerate(
                tqdm(self.train_dataloader, desc="train", total=self.len_epoch)
        ):
            for j, batch in enumerate(batchs):
                step += 1
                try:
                    batch = self.process_batch(
                        batch,
                        is_train=True,
                        metrics=self.train_metrics,
                    )
                except RuntimeError as e:
                    if "out of memory" in str(e) and self.skip_oom:
                        self.logger.warning("OOM on batch. Skipping batch.")
                        for p in self.model.parameters():
                            if p.grad is not None:
                                del p.grad  # free some memory
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
                self.train_metrics.update("grad norm", self.get_grad_norm())
                if step % self.log_step == 0:
                    self.writer.set_step(step)
                    self.logger.debug(
                        "Train Epoch: {} {} Loss: {:.6f}".format(
                            epoch, self._progress(batchs_idx), batch["total_loss"].item()
                        )
                    )
                    self.writer.add_scalar(
                        "learning rate", self.lr_scheduler.get_last_lr()[0]
                    )

                    self._log_scalars(self.train_metrics)
                    # we don't want to reset train metrics at the start of every epoch
                    # because we are interested in recent train metrics
                    last_train_metrics = self.train_metrics.result()
                    self.train_metrics.reset()
            if batchs_idx >= self.len_epoch:
                break

        log = last_train_metrics

        self._evaluation_epoch(epoch)

        return log

    def process_batch(self, batch, is_train: bool, metrics: MetricTracker):
        # Get Data
        character = batch["text"].long().to(self.device)
        mel_target = batch["mel_target"].float().to(self.device)
        duration = batch["duration"].int().to(self.device)
        energy = batch["energy"].float().to(self.device)
        pitch = batch["pitch"].float().to(self.device)
        mel_pos = batch["mel_pos"].long().to(self.device)
        src_pos = batch["src_pos"].long().to(self.device)
        max_mel_len = batch["mel_max_len"]

        if is_train:
            self.optimizer.zero_grad()

        # Forward
        mel_output, duration_output, pitch_out, energy_out = self.model(character,
                                                                        src_pos,
                                                                        mel_pos=mel_pos,
                                                                        mel_max_length=max_mel_len,
                                                                        length_target=duration,
                                                                        pitch_target=pitch,
                                                                        energy_target=energy)
        # Calc loss
        mel_loss, duration_loss, pitch_loss, energy_loss = self.loss(mel_output,
                                                                     duration_output,
                                                                     pitch_out,
                                                                     energy_out,
                                                                     mel_target,
                                                                     duration,
                                                                     pitch,
                                                                     energy)

        total_loss = mel_loss + duration_loss + pitch_loss + energy_loss

        t_l = total_loss.detach().cpu().numpy()
        m_l = mel_loss.detach().cpu().numpy()
        d_l = duration_loss.detach().cpu().numpy()
        p_l = pitch_loss.detach().cpu().numpy()
        e_l = energy_loss.detach().cpu().numpy()
        batch["duration_loss"] = d_l
        batch["mel_loss"] = m_l
        batch["total_loss"] = t_l
        batch["pitch_loss"] = p_l
        batch["energy_loss"] = e_l

        # Backward
        if is_train:
            total_loss.backward()
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
        metrics.update("loss", t_l)
        metrics.update("mel_loss", m_l)
        metrics.update("duration_loss", d_l)
        metrics.update("pitch_loss", p_l)
        metrics.update("energy_loss", e_l)
        return batch

    def _evaluation_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.writer.set_step(epoch * self.len_epoch, "eval")
        for i, phn in tqdm(enumerate(self.test_data)):
            wav = synthesis.synthesis(self.model, self.device, self.waveglow, phn)
            self._log_audio(f"result_{i}", wav)

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _log_audio(self, name: str, wav):
        self.writer.add_audio(name, wav, sample_rate=hparams_audio.sampling_rate)

    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))
