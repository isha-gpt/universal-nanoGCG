# universal_gcg.py
from typing import List, Tuple
import torch
from torch import Tensor
from tqdm import tqdm
import logging
from nanogcg.configs import GCGResult
from nanogcg.buffer import AttackBuffer
from nanogcg.gcg import GCG 
from nanogcg.utils import INIT_CHARS, find_executable_batch_size, filter_ids, sample_ids_from_grad, mellowmax
from transformers import set_seed
from scipy.stats import spearmanr
import csv
import os
import wandb
import threading
from threading import Lock

os.environ["TORCH_USE_CUDA_DSA"] = "1"

def _pad_and_concat(tensor_list: List[Tensor], pad_value: int = 0) -> Tuple[Tensor, List[int]]:
    """
    Given a list of tensors of shape (seq_len, dim), pad them along dim=0 into a batch.
    Returns a padded tensor of shape (batch, max_seq_len, dim) and a list of original lengths.
    """
    lengths = [t.size(0) for t in tensor_list]
    padded = torch.nn.utils.rnn.pad_sequence(tensor_list, batch_first=True, padding_value=pad_value)
    return padded, lengths

class UniversalGCG(GCG):
    """
    A subclass of GCG that implements multi-prompt optimization methods with batched computations.
    """
    
    def __init__(self, model, tokenizer, config, logger=None):
        super().__init__(model, tokenizer, config)
        self.logger = logger or logging.getLogger("nanogcg")
        
        self.jailbreak_log = config.jailbreak_log
        os.makedirs(os.path.dirname(self.jailbreak_log), exist_ok=True)
        with open(self.jailbreak_log, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["step", "loss", "suffix"]) 
    
    def _prepare_prompt_data(self, messages: List[str], targets: List[str]):
        """
        Prepare and embed tokens for each prompt/target pair.
        In addition to storing the original lists, we now pad the embeddings and token IDs
        to create batched tensors and record the true lengths.
        Also, if a dual model is used, we compute and batch its embeddings.
        """
        config = self.config
        model = self.model
        tokenizer = self.tokenizer
        
        assert len(messages) == len(targets)
        
        self.logger.info(f"Preparing {len(messages)} message-target pairs...")
    
        all_target_ids = []
        all_before_embeds = []
        all_after_embeds = []
        all_target_embeds = []
    
        if self.draft_model:
            assert self.draft_model and self.draft_tokenizer and self.draft_embedding_layer, "Draft model wasn't properly set up."
            all_draft_target_ids = []
            all_draft_before_embeds = []
            all_draft_after_embeds = []
            all_draft_target_embeds = []
        elif self.dual_model:
            assert self.dual_model and self.dual_embedding_layer, "Dual model wasn't properly set up."
            all_dual_target_ids = []
            all_dual_before_embeds = []
            all_dual_after_embeds = []
            all_dual_target_embeds = []
    
        # Also record raw token lengths for later use.
        before_lengths_list = []
        after_lengths_list = []
        target_lengths_list = []
    
        for message, target in zip(messages, targets):
            # messages = [{"role": "user", "content": message+"{optim_str}"}] 

            # template= tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            # if tokenizer.bos_token and template.startswith(tokenizer.bos_token):
            #     template = template.replace(tokenizer.bos_token, "")
            # template = message + "{optim_str}"
            
            # We do the formatting with the model's specific instruction template before passing the messages to the gcg class.

            before_str, after_str = message.split("{optim_str}")
            target = " " + target if config.add_space_before_target else target
    
            # Tokenize the parts that won't be optimized.
            before_ids = tokenizer([before_str], padding=False, return_tensors="pt")["input_ids"].to(model.device, torch.int64)
            after_ids = tokenizer([after_str], add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device, torch.int64)
            target_ids = tokenizer([target], add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device, torch.int64)
    
            # Embed the tokens using the primary model's embedding layer.
            embedding_layer = self.embedding_layer
            before_embeds = embedding_layer(before_ids)  # shape (1, L_before, emb_dim)
            after_embeds = embedding_layer(after_ids)    # shape (1, L_after, emb_dim)
            target_embeds = embedding_layer(target_ids)    # shape (1, L_target, emb_dim)
    
            all_target_ids.append(target_ids)
            all_before_embeds.append(before_embeds)
            all_after_embeds.append(after_embeds)
            all_target_embeds.append(target_embeds)
    
            before_lengths_list.append(before_ids.size(1))
            after_lengths_list.append(after_ids.size(1))
            target_lengths_list.append(target_ids.size(1))
    
            if self.draft_model:
                draft_before_ids = self.draft_tokenizer([before_str], padding=False, return_tensors="pt")["input_ids"].to(self.draft_model.device, torch.int64)
                draft_after_ids = self.draft_tokenizer([after_str], add_special_tokens=False, return_tensors="pt")["input_ids"].to(self.draft_model.device, torch.int64)
                draft_target_ids = self.draft_tokenizer([target], add_special_tokens=False, return_tensors="pt")["input_ids"].to(self.draft_model.device, torch.int64)
                all_draft_target_ids.append(draft_target_ids)
                all_draft_before_embeds.append(self.draft_embedding_layer(draft_before_ids))
                all_draft_after_embeds.append(self.draft_embedding_layer(draft_after_ids))
                all_draft_target_embeds.append(self.draft_embedding_layer(draft_target_ids))
            if self.dual_model:
                dual_before_ids = tokenizer([before_str], padding=False, return_tensors="pt")["input_ids"].to(self.dual_model.device, torch.int64)
                dual_after_ids = tokenizer([after_str], add_special_tokens=False, return_tensors="pt")["input_ids"].to(self.dual_model.device, torch.int64)
                dual_target_ids = tokenizer([target], add_special_tokens=False, return_tensors="pt")["input_ids"].to(self.dual_model.device, torch.int64)
                all_dual_target_ids.append(dual_target_ids)
                all_dual_before_embeds.append(self.dual_embedding_layer(dual_before_ids))
                all_dual_after_embeds.append(self.dual_embedding_layer(dual_after_ids))
                all_dual_target_embeds.append(self.dual_embedding_layer(dual_target_ids))
                
        # Save the first prompt info for initialization (unchanged).
        self.init_after_embeds = all_after_embeds[0]
        self.init_before_embeds = all_before_embeds[0]
        self.init_target_embeds = all_target_embeds[0]
        self.init_target_ids = all_target_ids[0]
    
        # Create batched (padded) versions for efficient computation for primary model.
        # Squeeze the batch dim from each embed (each is [1, L, D])
        self.batch_before_embeds, self.before_lengths = _pad_and_concat([x.squeeze(0) for x in all_before_embeds])
        self.batch_after_embeds, self.after_lengths = _pad_and_concat([x.squeeze(0) for x in all_after_embeds])
        self.batch_target_embeds, self.target_lengths = _pad_and_concat([x.squeeze(0) for x in all_target_embeds])
        # For target token IDs, pad using -100.
        self.batch_target_ids, _ = _pad_and_concat([x.squeeze(0) for x in all_target_ids], pad_value=-100)
    
        # --- New: Create batched versions for dual model (secondary) if applicable.
        if self.dual_model:
            self.batch_secondary_before_embeds, self.secondary_before_lengths = _pad_and_concat(
                [x.squeeze(0) for x in all_dual_before_embeds]
            )
            self.batch_secondary_after_embeds, self.secondary_after_lengths = _pad_and_concat(
                [x.squeeze(0) for x in all_dual_after_embeds]
            )
            self.batch_secondary_target_embeds, self.secondary_target_lengths = _pad_and_concat(
                [x.squeeze(0) for x in all_dual_target_embeds]
            )
            self.batch_secondary_target_ids, _ = _pad_and_concat(
                [x.squeeze(0) for x in all_dual_target_ids], pad_value=-100
            )
    
        # Return the original lists for backward compatibility if required.
        if self.draft_model:
            return (all_target_ids, all_before_embeds, all_after_embeds, all_target_embeds,
                    all_draft_target_ids, all_draft_before_embeds, all_draft_after_embeds, all_draft_target_embeds)
        elif self.dual_model:
            return (all_target_ids, all_before_embeds, all_after_embeds, all_target_embeds,
                    all_dual_target_ids, all_dual_before_embeds, all_dual_after_embeds, all_dual_target_embeds)
        else:
            return (all_target_ids, all_before_embeds, all_after_embeds, all_target_embeds)
            
    def init_buffer_multi_prompt(self) -> AttackBuffer:
        model = self.model
        tokenizer = self.tokenizer
        config = self.config
    
        self.logger.info(f"Initializing attack buffer of size {config.buffer_size}...")
    
        buffer = AttackBuffer(config.buffer_size)
    
        if isinstance(config.optim_str_init, str):
            init_optim_ids = tokenizer(
                config.optim_str_init, add_special_tokens=False, return_tensors="pt"
            )["input_ids"].to(model.device)
            if config.buffer_size > 1:
                init_buffer_ids = tokenizer(
                    INIT_CHARS, add_special_tokens=False, return_tensors="pt"
                )["input_ids"].squeeze().to(model.device)
                init_indices = torch.randint(
                    0, init_buffer_ids.shape[0], (config.buffer_size - 1, init_optim_ids.shape[1])
                )
                init_buffer_ids = torch.cat([init_optim_ids, init_buffer_ids[init_indices]], dim=0)
            else:
                init_buffer_ids = init_optim_ids
        else:  # assume list
            if len(config.optim_str_init) != config.buffer_size:
                self.logger.warning(
                    f"Using {len(config.optim_str_init)} initializations but buffer size is set to {config.buffer_size}"
                )
            try:
                init_buffer_ids = tokenizer(
                    config.optim_str_init, add_special_tokens=False, return_tensors="pt"
                )["input_ids"].to(model.device)
            except ValueError:
                self.logger.error("Unable to create buffer. Ensure that all initializations tokenize to the same length.")
    
        true_buffer_size = max(1, config.buffer_size)
    
        if self.prefix_cache:
            init_buffer_embeds = torch.cat([
                self.embedding_layer(init_buffer_ids),
                self.init_after_embeds.repeat(true_buffer_size, 1, 1),
                self.init_target_embeds.repeat(true_buffer_size, 1, 1),
            ], dim=1)
        else:
            init_buffer_embeds = torch.cat([
                self.init_before_embeds.repeat(true_buffer_size, 1, 1),
                self.embedding_layer(init_buffer_ids),
                self.init_after_embeds.repeat(true_buffer_size, 1, 1),
                self.init_target_embeds.repeat(true_buffer_size, 1, 1),
            ], dim=1)
            
        self.logger.debug(f"init_buffer_embeds shape: {init_buffer_embeds.shape}")
        self.logger.debug(f"init_buffer_embeds sample: {init_buffer_embeds[0][:5]}")
    
        self.target_ids = self.init_target_ids
        init_buffer_losses = find_executable_batch_size(self._compute_candidates_loss_original, true_buffer_size)(init_buffer_embeds)
    
        self.logger.debug(f"init_buffer_losses shape: {init_buffer_losses.shape}")
        self.logger.debug(f"init_buffer_losses values: {init_buffer_losses}")
    
        for i in range(true_buffer_size):
            buffer.add(init_buffer_losses[i], init_buffer_ids[[i]])
    
        buffer.log_buffer(tokenizer)
    
        self.logger.debug("Initialized attack buffer.")
    
        return buffer
    
    
    def compute_token_gradient_multi(self, optim_ids: Tensor, num_contexts) -> Tensor:
        """
        Computes the gradient of the aggregated loss (averaged over all prompt contexts)
        with respect to the one-hot candidate token matrix for both the primary and secondary models,
        and then aggregates them (averaging).
        """
        # -------------------- Primary Branch --------------------
        primary_model = self.model
        primary_embedder = self.embedding_layer
        # Create one-hot encoding for candidate tokens for primary branch.
        optim_ids_onehot_primary = torch.nn.functional.one_hot(
            optim_ids, num_classes=primary_embedder.num_embeddings
        )
        optim_ids_onehot_primary = optim_ids_onehot_primary.to(primary_model.device, primary_model.dtype)
        optim_ids_onehot_primary.requires_grad_()

        # Get candidate embeddings from primary embedding layer.
        optim_embeds_primary = optim_ids_onehot_primary @ primary_embedder.weight  # (1, n_optim, emb_dim_primary)
        optim_embeds_batch_primary = optim_embeds_primary.expand(num_contexts, -1, -1)

        batched_sequences_primary = []
        batched_target_labels_primary = []
        for i in range(num_contexts):
            if self.prefix_cache:
                # Sequence: [candidate, after, target]
                seq = torch.cat([
                    optim_embeds_batch_primary[i],
                    self.batch_after_embeds[i, :self.after_lengths[i]],
                    self.batch_target_embeds[i, :self.target_lengths[i]]
                ], dim=0)
                L_optim = optim_embeds_batch_primary.size(1)
                start = L_optim + self.after_lengths[i] - 1
            else:
                # Sequence: [before, candidate, after, target]
                seq = torch.cat([
                    self.batch_before_embeds[i, :self.before_lengths[i]],
                    optim_embeds_batch_primary[i],
                    self.batch_after_embeds[i, :self.after_lengths[i]],
                    self.batch_target_embeds[i, :self.target_lengths[i]]
                ], dim=0)
                L_before = self.before_lengths[i]
                L_optim = optim_embeds_batch_primary.size(1)
                start = L_before + L_optim + self.after_lengths[i] - 1
            batched_sequences_primary.append(seq)
            labels = torch.full((seq.size(0),), -100, dtype=torch.long, device=seq.device)
            labels[start:start+self.target_lengths[i]] = self.batch_target_ids[i, :self.target_lengths[i]]
            batched_target_labels_primary.append(labels)
        padded_seq_primary = torch.nn.utils.rnn.pad_sequence(batched_sequences_primary, batch_first=True)
        padded_labels_primary = torch.nn.utils.rnn.pad_sequence(batched_target_labels_primary, batch_first=True, padding_value=-100)

        outputs_primary = primary_model(inputs_embeds=padded_seq_primary)
        logits_primary = outputs_primary.logits
        loss_primary = torch.nn.functional.cross_entropy(
            logits_primary.view(-1, logits_primary.size(-1)),
            padded_labels_primary.view(-1),
            ignore_index=-100,
            reduction='mean'
        )
        grad_primary = torch.autograd.grad(loss_primary, optim_ids_onehot_primary)[0]

        # -------------------- Secondary (Dual) Branch --------------------
        dual_model = self.dual_model
        if dual_model:
            dual_embedder = self.dual_embedding_layer
            # Create one-hot encoding for candidate tokens for secondary branch.
            optim_ids_onehot_secondary = torch.nn.functional.one_hot(
                optim_ids, num_classes=dual_embedder.num_embeddings
            )
            optim_ids_onehot_secondary = optim_ids_onehot_secondary.to(dual_model.device, dual_model.dtype)
            optim_ids_onehot_secondary.requires_grad_()

            # Get candidate embeddings from dual embedding layer.
            optim_embeds_secondary = optim_ids_onehot_secondary @ dual_embedder.weight  # (1, n_optim, emb_dim_secondary)
            optim_embeds_batch_secondary = optim_embeds_secondary.expand(num_contexts, -1, -1)

            batched_sequences_secondary = []
            batched_target_labels_secondary = []
            for i in range(num_contexts):
                if self.prefix_cache:
                    # Sequence: [candidate, dual after, dual target]
                    seq = torch.cat([
                        optim_embeds_batch_secondary[i],
                        self.batch_secondary_after_embeds[i, :self.secondary_after_lengths[i]],
                        self.batch_secondary_target_embeds[i, :self.secondary_target_lengths[i]]
                    ], dim=0)
                    L_optim = optim_embeds_batch_secondary.size(1)
                    start = L_optim + self.secondary_after_lengths[i] - 1
                else:
                    # Sequence: [dual before, candidate, dual after, dual target]
                    seq = torch.cat([
                        self.batch_secondary_before_embeds[i, :self.secondary_before_lengths[i]],
                        optim_embeds_batch_secondary[i],
                        self.batch_secondary_after_embeds[i, :self.secondary_after_lengths[i]],
                        self.batch_secondary_target_embeds[i, :self.secondary_target_lengths[i]]
                    ], dim=0)
                    L_before = self.secondary_before_lengths[i]
                    L_optim = optim_embeds_batch_secondary.size(1)
                    start = L_before + L_optim + self.secondary_after_lengths[i] - 1
                batched_sequences_secondary.append(seq)
                labels = torch.full((seq.size(0),), -100, dtype=torch.long, device=seq.device)
                labels[start:start+self.secondary_target_lengths[i]] = self.batch_secondary_target_ids[i, :self.secondary_target_lengths[i]]
                batched_target_labels_secondary.append(labels)
            padded_seq_secondary = torch.nn.utils.rnn.pad_sequence(batched_sequences_secondary, batch_first=True)
            padded_labels_secondary = torch.nn.utils.rnn.pad_sequence(batched_target_labels_secondary, batch_first=True, padding_value=-100)

            outputs_secondary = dual_model(inputs_embeds=padded_seq_secondary)
            logits_secondary = outputs_secondary.logits
            loss_secondary = torch.nn.functional.cross_entropy(
                logits_secondary.view(-1, logits_secondary.size(-1)),
                padded_labels_secondary.view(-1),
                ignore_index=-100,
                reduction='mean'
            )
            grad_secondary = torch.autograd.grad(loss_secondary, optim_ids_onehot_secondary)[0]

            # -------------------- Aggregate Gradients --------------------
            # Move secondary gradient to primary device and average.
            grad_secondary = grad_secondary.to(primary_model.device)
            grad_total = (grad_primary + grad_secondary) / 2
            return grad_total

        
        else:
            return grad_primary

    
    
    def _build_candidate_batch(self, sampled_ids: Tensor, m_c: int, dual=False):
        """
        Given the sampled candidate token IDs (shape: [num_candidates, token_length]),
        builds batched candidate sequences (with prompt-specific embeddings) and corresponding
        target label tensors (padded with -100) for loss computation.
        Returns:
            candidate_seqs: Tensor of shape (m_c, num_candidates, max_seq_len, emb_dim)
            target_info: dict containing the padded target label tensor.
        """
        num_candidates = sampled_ids.shape[0]
        candidate_seqs = []
        candidate_labels = []
        
        if dual:
            embedder = self.dual_embedding_layer
            dual_sampled_ids = sampled_ids.to(self.dual_model.device)
        else:
            embedder = self.embedding_layer
        
        for i in range(m_c):
            seqs = []
            labels = []
            # Use dual prompt embeddings if dual==True
            if dual:
                before_embeds = self.batch_secondary_before_embeds[i, :self.secondary_before_lengths[i]].to(self.dual_model.device)
                after_embeds  = self.batch_secondary_after_embeds[i, :self.secondary_after_lengths[i]].to(self.dual_model.device)
                target_embeds = self.batch_secondary_target_embeds[i, :self.secondary_target_lengths[i]].to(self.dual_model.device)
                target_ids    = self.batch_secondary_target_ids[i, :self.secondary_target_lengths[i]].to(self.dual_model.device)
            else:
                before_embeds = self.batch_before_embeds[i, :self.before_lengths[i]]
                after_embeds  = self.batch_after_embeds[i, :self.after_lengths[i]]
                target_embeds = self.batch_target_embeds[i, :self.target_lengths[i]]
                target_ids    = self.batch_target_ids[i, :self.target_lengths[i]]
        
            for j in range(num_candidates):
                if dual:
                    optim_emb = embedder(dual_sampled_ids[j].unsqueeze(0))
                else:
                    optim_emb = embedder(sampled_ids[j].unsqueeze(0))
                optim_emb_squeezed = optim_emb.squeeze(0)
        
                if self.prefix_cache:
                    seq = torch.cat([
                        optim_emb_squeezed,
                        after_embeds,
                        target_embeds
                    ], dim=0)
                    L_optim = optim_emb.size(1)
                    start = L_optim + after_embeds.size(0) - 1
                else:
                    seq = torch.cat([
                        before_embeds,
                        optim_emb_squeezed,
                        after_embeds,
                        target_embeds
                    ], dim=0)
                    L_before = before_embeds.size(0)
                    L_optim = optim_emb.size(1)
                    start = L_before + L_optim + after_embeds.size(0) - 1
        
                seqs.append(seq)
                label = torch.full((seq.size(0),), -100, dtype=torch.long, device=seq.device)
                label[start:start+target_ids.size(0)] = target_ids
                labels.append(label)
        
            # --- Intra-Prompt Padding ---
            prompt_max_seq_len = max([s.size(0) for s in seqs])
            padded_seqs = [torch.nn.functional.pad(s, (0, 0, 0, prompt_max_seq_len - s.size(0)), value=0)
                        if s.size(0) < prompt_max_seq_len else s for s in seqs]
            prompt_max_label_len = max([l.size(0) for l in labels])
            padded_labels = [torch.nn.functional.pad(l, (0, prompt_max_label_len - l.size(0)), value=-100)
                            if l.size(0) < prompt_max_label_len else l for l in labels]
        
            prompt_seqs = torch.stack(padded_seqs, dim=0)
            prompt_labels = torch.stack(padded_labels, dim=0)
            candidate_seqs.append(prompt_seqs)
            candidate_labels.append(prompt_labels)
        
        # --- Cross-Prompt Padding (if necessary) ---
        final_max_seq_len = max([t.size(1) for t in candidate_seqs])
        for idx, t in enumerate(candidate_seqs):
            if t.size(1) < final_max_seq_len:
                pad_size = final_max_seq_len - t.size(1)
                candidate_seqs[idx] = torch.nn.functional.pad(t, (0, 0, 0, pad_size), value=0)
        final_max_label_len = max([t.size(1) for t in candidate_labels])
        for idx, t in enumerate(candidate_labels):
            if t.size(1) < final_max_label_len:
                pad_size = final_max_label_len - t.size(1)
                candidate_labels[idx] = torch.nn.functional.pad(t, (0, pad_size), value=-100)
        
        candidate_seqs = torch.stack(candidate_seqs, dim=0)
        candidate_labels = torch.stack(candidate_labels, dim=0)
        self.logger.debug(f"Final candidate_seqs shape: {candidate_seqs.shape}")
        target_info = {'labels': candidate_labels}
        return candidate_seqs, target_info

    def _compute_candidates_loss_multi_prompt(
        self,
        search_batch_size: int,
        candidate_seqs: Tensor,   # shape: (num_prompts, num_candidates, seq_len, emb_dim)
        target_info: dict,
        dual: bool = False
    ) -> Tensor:
        model = self.dual_model if dual else self.model
        num_prompts, num_candidates, seq_len, emb_dim = candidate_seqs.shape
        # Flatten candidates: shape (num_prompts*num_candidates, seq_len, emb_dim)
        flattened = candidate_seqs.view(num_prompts * num_candidates, seq_len, emb_dim)
        # Flatten target labels accordingly.
        flattened_target_labels = target_info['labels'].view(num_prompts * num_candidates, seq_len)
        losses = []
        for i in range(0, flattened.size(0), search_batch_size):
            batch = flattened[i:i + search_batch_size]
            # Slice target labels for this candidate batch.
            batch_target_labels = flattened_target_labels[i:i + search_batch_size]
            with torch.no_grad():
                if self.prefix_cache:
                    outputs = model(inputs_embeds=batch, past_key_values=self.prefix_cache, use_cache=True)
                else:
                    outputs = model(inputs_embeds=batch)
                logits = outputs.logits
                loss_batch = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    batch_target_labels.view(-1),
                    ignore_index=-100,
                    reduction='none'
                )
                loss_batch = loss_batch.view(batch.size(0), seq_len)
                candidate_losses = []
                for j in range(batch.size(0)):
                    valid = (batch_target_labels[j] != -100)
                    if valid.sum() > 0:
                        candidate_losses.append(loss_batch[j][valid].mean())
                    else:
                        candidate_losses.append(torch.tensor(0., device=loss_batch.device))
                losses.append(torch.stack(candidate_losses))
        losses_tensor = torch.cat(losses)  # This should now have num_prompts*num_candidates elements.
        losses_tensor = losses_tensor.view(num_prompts, num_candidates)
        return losses_tensor

    def run_multi_prompt(
        self,
        messages: List[str],
        targets: List[str],
    ) -> GCGResult:
        
        tokenizer = self.tokenizer
        config = self.config

        assert not (config.dual_model and config.probe_sampling_config), "Cannot run with both dual model and probe sampling config."
        
        if config.seed is not None:
            set_seed(config.seed)
            torch.use_deterministic_algorithms(True, warn_only=True)
    
        if not isinstance(messages, list):
            raise ValueError("Messages must be in the form of a list of strings.")
        
        self.logger.info("🔹 Processing prompts & targets for optimization...")
        prep_result = self._prepare_prompt_data(messages, targets)
        # Store the original lists if needed for dual/draft operations.
        self.all_target_ids, self.all_before_embeds, self.all_after_embeds, self.all_target_embeds, *rest = prep_result
        if rest:
            self.secondary_target_ids, self.secondary_before_embeds, self.secondary_after_embeds, self.secondary_target_embeds = rest
        else:
            self.secondary_target_ids = self.secondary_before_embeds = self.secondary_after_embeds = self.secondary_target_embeds = None
        
        self.logger.info("✅ Prompt & target processing complete.")
        
        buffer = self.init_buffer_multi_prompt()
        optim_ids = buffer.get_best_ids()

        results = []

        if config.incremental:
            outer = len(messages)
            inner = config.num_steps // outer
            m_c = 0
        else:
            outer = 1
            inner = config.num_steps
            m_c = len(messages)
        
        for index in range(outer):
            if config.incremental:
                m_c += 1
                # Only prepare data for current subset of messages
                current_messages = messages[:m_c]
                current_targets = targets[:m_c]
                prep_result = self._prepare_prompt_data(current_messages, current_targets)
                self.all_target_ids, self.all_before_embeds, self.all_after_embeds, self.all_target_embeds, *rest = prep_result
                if rest:
                    self.secondary_target_ids, self.secondary_before_embeds, self.secondary_after_embeds, self.secondary_target_embeds = rest
                else:
                    self.secondary_target_ids = self.secondary_before_embeds = self.secondary_after_embeds = self.secondary_target_embeds = None
                self.batch_before_embeds = self.batch_before_embeds[:m_c]
                self.before_lengths = self.before_lengths[:m_c]
                self.batch_after_embeds = self.batch_after_embeds[:m_c]
                self.after_lengths = self.after_lengths[:m_c]
                self.batch_target_embeds = self.batch_target_embeds[:m_c]
                self.target_lengths = self.target_lengths[:m_c]
                self.batch_target_ids = self.batch_target_ids[:m_c]

                if self.dual_model:
                    self.batch_secondary_before_embeds = self.batch_secondary_before_embeds[:m_c]
                    self.secondary_before_lengths = self.secondary_before_lengths[:m_c]
                    self.batch_secondary_after_embeds = self.batch_secondary_after_embeds[:m_c]
                    self.secondary_after_lengths = self.secondary_after_lengths[:m_c]
                    self.batch_secondary_target_embeds = self.batch_secondary_target_embeds[:m_c]
                    self.secondary_target_lengths = self.secondary_target_lengths[:m_c]
                    self.batch_secondary_target_ids = self.batch_secondary_target_ids[:m_c]

            else:
                current_messages = messages
                current_targets = targets
            
            self.logger.info(f"\n🚀 Starting optimization for {m_c} message(s)...")
            losses = []
            optim_strings = []
            for step in tqdm(range(inner), desc="Optimizing Suffix"):
                self.logger.info(f"\n🔄 Step {step + 1}/{config.num_steps} - Computing token gradient...")
                optim_ids_onehot_grad = self.compute_token_gradient_multi(optim_ids, m_c)
    
                self.logger.debug(f"🔍 Gradient Shape: {optim_ids_onehot_grad.shape}")
                self.logger.debug(f"🔍 Optim IDs Before Sampling: {optim_ids}")
    
                with torch.no_grad():
                    sampled_ids = sample_ids_from_grad(
                        optim_ids.squeeze(0),
                        optim_ids_onehot_grad.squeeze(0),
                        config.search_width,
                        config.topk,
                        config.n_replace,
                        not_allowed_ids=self.not_allowed_ids,
                    )
    
                    if config.filter_ids:
                        sampled_ids = filter_ids(sampled_ids, tokenizer)
    
                    new_search_width = sampled_ids.shape[0]
                    batch_size = new_search_width if config.batch_size is None else config.batch_size
    
                    # Build batched candidate sequences and target labels for all prompts.
                    candidate_seqs, target_info = self._build_candidate_batch(sampled_ids, m_c)
    
    
                    if self.config.probe_sampling_config is not None:
                        self.logger.info("🧮 Computing losses with draft model...")
                        # (Probe sampling branch would be updated similarly using batched candidates.)
                        current_loss, optim_ids = find_executable_batch_size(
                            self._compute_candidates_loss_probe_sampling_multi, batch_size
                        )(candidate_seqs, sampled_ids, 
                          self.secondary_before_embeds[:m_c], self.secondary_after_embeds[:m_c],
                          self.secondary_target_embeds[:m_c], self.all_target_ids[:m_c], self.secondary_target_ids[:m_c])
                    elif self.dual_model is not None:
                        self.logger.info("🧮 Computing losses with dual model...")
                        loss_results = {}
                        loss_lock = Lock()

                        def compute_primary():
                            primary_loss = find_executable_batch_size(self._compute_candidates_loss_multi_prompt, batch_size)(candidate_seqs, target_info)
                            with loss_lock:
                                loss_results['primary'] = primary_loss

                        def compute_dual():
                            dual_candidate_seqs, dual_target_info = self._build_candidate_batch(sampled_ids, m_c, dual=True)
                            dual_loss = find_executable_batch_size(self._compute_candidates_loss_multi_prompt, batch_size)(
                                dual_candidate_seqs, dual_target_info, dual=True
                            )
                            with loss_lock:
                                loss_results['dual'] = dual_loss

                        # Create and start threads for each computation.
                        thread_primary = threading.Thread(target=compute_primary)
                        thread_dual = threading.Thread(target=compute_dual)

                        thread_primary.start()
                        thread_dual.start()

                        # Wait for both threads to complete.
                        thread_primary.join()
                        thread_dual.join()

                        # Retrieve the computed losses.
                        losses_tensor = loss_results['primary']
                        dual_losses_tensor = loss_results['dual']

                        # Combine the losses (here by averaging, adjust as needed).
                        combined_losses_tensor = (losses_tensor + dual_losses_tensor.to(losses_tensor.device)) / 2
                        avg_losses = combined_losses_tensor.mean(dim=0)

                        self.logger.debug(f"🔎 Main model losses: shape: {losses_tensor.shape}, first 10: {losses_tensor.flatten()[:10]}")
                        self.logger.debug(f"🔎 Dual model losses: shape: {dual_losses_tensor.shape}, first 10: {dual_losses_tensor.flatten()[:10]}")
                        self.logger.debug(f"🔎 Avg Losses: shape: {avg_losses.shape}, first 10: {avg_losses.flatten()[:10]}")

                        current_loss = avg_losses.min().item()
                        best_idx = avg_losses.argmin().item()
                        optim_ids = sampled_ids[best_idx].unsqueeze(0)

                    else:
                        self.logger.info("🧮 Computing Losses...")
                        losses_tensor = find_executable_batch_size(
                            self._compute_candidates_loss_multi_prompt, batch_size
                        )(candidate_seqs, target_info)
                        avg_losses = losses_tensor.mean(dim=0)
                        current_loss = avg_losses.min().item()
                        best_idx = avg_losses.argmin().item()
                        optim_ids = sampled_ids[best_idx].unsqueeze(0)
    
                    self.logger.info(f"✅ Loss: {current_loss}")
                    losses.append(current_loss)
                    wandb.log({"Loss": current_loss})
    
                    if buffer.size == 0 or current_loss < buffer.get_highest_loss():
                        buffer.add(current_loss, optim_ids)
    
                optim_ids = buffer.get_best_ids()
                optim_str = tokenizer.batch_decode(optim_ids)[0]
                optim_strings.append(optim_str)
    
                self.logger.info(f"🔹 Current Best Optimized String: {optim_str}")
    
                if self.stop_flag:
                    self.logger.info("⚠️ Early stopping triggered due to finding a perfect match.")
                    break
                
                if step % 10 == 0:
                    with open(self.jailbreak_log, "a", newline="") as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([step, min(losses) if losses else None, optim_strings[-1] if optim_strings else ""])
    
            min_loss_index = losses.index(min(losses))
    
            result = GCGResult(
                best_loss=losses[min_loss_index],
                best_string=optim_strings[min_loss_index],
                losses=losses,
                strings=optim_strings,
            )
            results.append(result)

            # step_number = index * inner + min_loss_index
    
        self.logger.info("\n✅ **Suffix Optimization Completed Successfully!**")
        return results
